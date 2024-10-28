import typing
import pytorch_lightning as pl
import torch
import torchmetrics.functional as tf
import wandb
from datasets.SCEMILA.SCEMILA_lightning_wrapper import SCEMILA
from datasets.SCEMILA.SEMILA_indexer import SCEMILA_Indexer
from experiments.SCEMILA_experiment import DataMatrix, SCEMILA_Experiment
from models.SCEMILA.topo_SCEMILA_model import TopoAMiL
from models.model_factory import get_module
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from trainer_scripts.label_smoothing_scheduler import LabelSmoothingScheduler

class TopoScheduler:
    DEFAULT = 0.0
    def __init__(self,experiment: pl.LightningModule = None,dict_args:dict={}):
        sched_type = dict_args.get("type", "constant")  # Default to empty string if "type" not found
        self.type = sched_type
        self.experiment = experiment
        self.args = dict_args

        if self.type == "constant":
            self.function = partial(self.constant_fnc, lam=dict_args.get("lam", 0.0005))  # Default to None if "lam" not found
        elif self.type == "exp_epoch":
            self.function = partial(
                self.exp_schedule,
                lam_topo=dict_args.get("lam", 0.000025),  # Default to None if "lam_topo" not found
                lam_topo_per_epoch_decay=dict_args.get("lam_topo_per_epoch_decay", 1.008),  # Default to None if "lam_topo_per_epoch_decay" not found
                max_lam=dict_args.get("max_lam", float('inf')),  # Default to inf if "max_lam" not found
                min_lam=dict_args.get("min_lam", -float('inf'))  # Default to -inf if "min_lam" not found
            )
        elif self.type == "on_off":
            self.function = partial(
                self.on_off,
                tracked_metric=dict_args.get("tracked_metric", "train_correct_epoch"),  # Default to None if "tracked_metric" not found
                metric_threshold=dict_args.get("metric_threshold", 0.95),  # Default to None if "metric_threshold" not found
                lam_high=dict_args.get("lam_high", 0.001),  # Default to None if "lam_high" not found
                lam_low=dict_args.get("lam_low", 0.0),  # Default to None if "lam_low" not found
                stay_on=dict_args.get("stay_on",True),
                active_high=dict_args.get("active_high", True)
            )
        elif self.type == "match_mil_loss":
            self.function = partial(
                self.match_mil_loss_with_trigger,
                tracked_metric=dict_args.get("tracked_metric", "train_correct_epoch"),  # Default to None if "tracked_metric" not found
                metric_threshold=dict_args.get("metric_threshold", 0.95),  # Default to None if "metric_threshold" not found
                active_high=dict_args.get("active_high", True),  # Default to None if "active_high" not found
                match_on_step=dict_args.get("match_on_step", True),  # If true match loss on a step level, if false matches it on an epoch level
                default_value=dict_args.get("default_value", TopoScheduler.DEFAULT)
            )
        else:
            raise ValueError("This type of Topo scheduling is not supported")
    

    def __call__(self, step_metrics):
        return self.function(step_metrics=step_metrics)

    def constant_fnc(self,step_metrics, lam):
        return lam
    
    def on_off(self,step_metrics, tracked_metric, metric_threshold, lam_high, lam_low,stay_on,active_high):
        if not hasattr(self,"stay_on"):
            self.stay_on = stay_on
            self.activated = False
        metric = self.experiment.trainer.callback_metrics.get(tracked_metric,None)
        if metric is not None:
            if ((metric > metric_threshold) == active_high) or self.activated:
                if self.stay_on:
                    self.activated = True
                return lam_high
            else:
                return lam_low
        elif self.experiment.current_epoch == 0:
            return lam_low
        else:
            print(f"WARNING: Could not find metric {tracked_metric} reverting to default value {TopoScheduler.DEFAULT}")
            return TopoScheduler.DEFAULT #if user requested an on_epoch metric, then it is not accessable during first epoch
        
    def match_mil_loss_with_trigger(self, step_metrics, tracked_metric, metric_threshold,default_value,match_on_step,active_high):
        metric = self.experiment.trainer.callback_metrics.get(tracked_metric,None) 
        if metric is not None:
            if match_on_step:
                if (metric > metric_threshold) == active_high:
                    with torch.no_grad():
                        result = step_metrics["mil_loss"] / step_metrics["topo_loss"]
                    return result
                else:
                    return default_value
            else:
                if (metric > metric_threshold) == active_high:
                    with torch.no_grad():
                        result = self.experiment.trainer.callback_metrics.get("train_mil_loss_epoch",0.0)  / self.experiment.trainer.callback_metrics.get("train_topo_loss_epoch",1.0) 
                    return result
                else:
                    return default_value
                
        else:
            return TopoScheduler.DEFAULT #if user requested an on_epoch metric, then it is not accessable during first epoch

    def exp_schedule(self,step_metrics,lam_topo,lam_topo_per_epoch_decay,max_lam,min_lam):
        return max(min(lam_topo * (lam_topo_per_epoch_decay**self.experiment.current_epoch),max_lam),min_lam)

class TopoSCEMILA_Experiment(pl.LightningModule):
    def __init__(self, model: TopoAMiL, params: typing.Dict[str, typing.Any]) -> None:
        super(TopoSCEMILA_Experiment, self).__init__()
        self.model = model
        self.params = params
        self.n_c = self.params.get("num_class",5)
        self.class_weighting_factor = self.params.get("class_weighting_factor",0.0)
        self.curr_device = None
        self.dataset = None
        self.current_data_object = DataMatrix()
        self.label_smoothing_settings = self.params.get("label_smoothing",{"smoothing":0.0,"per_epoch_decay":1.0,"train_correct_threshold":1.01})#keep smoothing on all the time >1.0        
        self.train_confusion_matrix,self.val_confusion_matrix,self.test_confusion_matrix = torch.zeros(self.n_c, self.n_c),torch.zeros(self.n_c, self.n_c),torch.zeros(self.n_c, self.n_c),
        self.test_metrics = []
        self.label_smoothing_scheduler = LabelSmoothingScheduler(experiment=self,**self.label_smoothing_settings)
        self.model.set_mil_smoothing(self.label_smoothing_scheduler.get_current_smoothing())
        self.indexer = SCEMILA_Indexer.get_indexer()
        if self.class_weighting_factor:
            self.class_weights = self.get_class_weights(self.class_weighting_factor)
        self.topo_scheduler = TopoScheduler(self,params.get("topo_scheduler",{}))
        self.kill_mil_loss = params.get("kill_mil_loss",False)
        pass

    def set_dataset_for_latent_visualization(self,dataset):
        assert isinstance(dataset,SCEMILA)
        self.dataset = dataset

    def get_class_weights(self,weighting_factor):
        assert 0 <= weighting_factor <= 1
        train_class_distribution = self.indexer.class_sample_counts_in_train_set
        assert len(train_class_distribution) == self.n_c
        total_number = sum([sample_count for class_name,sample_count in train_class_distribution.items()])
        class_weights = {self.indexer.convert_from_int_to_label_bag_level(class_key):torch.tensor((1 - weighting_factor)+ weighting_factor*total_number/(self.n_c*class_count)) for class_key,class_count in train_class_distribution.items()}
        return class_weights
    

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)

    def convert_matrix_to_str(self, mat):
        return str(mat.tolist())
    
    def log_confusion_matrix(self,phase,confusion_matrix):
        try:
            plt.figure(figsize=(10, 7))
            labels = [self.indexer.convert_from_int_to_label_bag_level(i) for i in range(self.model.class_count)]
            sns.heatmap(confusion_matrix.cpu().numpy(), annot=True, fmt="g", cmap="Blues",xticklabels = labels,yticklabels=labels,cbar=False)
            plt.xlabel("Predicted labels")
            plt.ylabel("True labels")
            plt.title("Confusion Matrix")
            wandb.log({f"{phase} confusion matrix": wandb.Image(plt)})
            plt.close()
        except Exception as e:
            print(f"Could not upload the {phase} {self.current_epoch}th confusion matrix")



    def training_step(self, batch, batch_idx):
        bag, bag_label, dist_mat = batch
        model_output = self.model(bag)
        train_joint_loss = self.model.mil_loss_function(model_output[0], bag_label[0])
        train_step_topo_loss = self.model.topo_loss_function(
            latent_distance_matrix=model_output[-1], input_distance_matrix=dist_mat
        )
        train_joint_loss.update(train_step_topo_loss)
        train_joint_loss["topo_weight_loss_epoch"] = self.topo_scheduler(train_joint_loss)
        if self.kill_mil_loss:
            train_joint_loss["loss"] = train_joint_loss["topo_weight_loss_epoch"] * train_joint_loss["topo_loss"]
        else:
            train_joint_loss["loss"] = (
                train_joint_loss["mil_loss"] + train_joint_loss["topo_weight_loss_epoch"] * train_joint_loss["topo_loss"]
            )        
        self.train_confusion_matrix[
            train_joint_loss["label"], train_joint_loss["prediction_int"]
        ] += int(1)
        train_joint_loss["LR"] = self.trainer.optimizers[0].param_groups[0]['lr']
        if self.class_weighting_factor:
            train_joint_loss["loss"] = train_joint_loss["loss"] * self.class_weights[train_joint_loss["label"]]
        self.log_step(train_joint_loss, "train", progress_bar= True)
        return train_joint_loss
            
    def log_step(self,step_loss,phase,progress_bar = False):
        self.log(f"{phase}_loss",step_loss["loss"].data,on_step=True,on_epoch= True,prog_bar= progress_bar,logger = True)
        self.log(f"{phase}_correct",step_loss["correct"],on_step=True,on_epoch= True,prog_bar= progress_bar,logger = True)
        self.log(f"{phase}_mil_loss",step_loss["mil_loss"].data,on_step=True,on_epoch= True,prog_bar= progress_bar,logger = True)
        self.log(f"{phase}_topo_loss",step_loss["topo_loss"].data,on_step=True,on_epoch= True,prog_bar= progress_bar,logger = True)
        if "topo_step_log" in step_loss:
            topo_log = step_loss["topo_step_log"]
            for ending in ["_1_on_2","_2_on_1"]:
                if f"topo_time_taken{ending}" in topo_log and phase == "train":
                    self.log(f"per_step_topo_calc_time{ending}_topo_analytics",topo_log[f"topo_time_taken{ending}"],on_step=False,on_epoch= True,prog_bar= False,logger = True)
                if f"nb_of_persistent_edges{ending}" in topo_log and phase == "train":
                    self.log(f"persistent_edges_nb{ending}_topo_analytics",topo_log[f"nb_of_persistent_edges{ending}"],on_step=False,on_epoch= True,prog_bar= False,logger = True)
                if f"percentage_toporeg_calc{ending}" in topo_log and phase == "train":
                    self.log(f"percentage_toporeg_calculated{ending}_topo_analytics",topo_log[f"percentage_toporeg_calc{ending}"],on_step=False,on_epoch= True,prog_bar= False,logger = True)
                if f"pull_push_ratio{ending}" in topo_log and phase == "train":
                    self.log(f"pull_to_push_topo_ratio{ending}_topo_analytics",topo_log[f"pull_push_ratio{ending}"],on_step=False,on_epoch= True,prog_bar= False,logger = True)
                if f"nb_pairwise_distance_influenced{ending}" in topo_log and phase == "train":
                    self.log(f"pairwise_distance_influenced_nb{ending}_topo_analytics",topo_log[f"nb_pairwise_distance_influenced{ending}"],on_step=False,on_epoch= True,prog_bar= False,logger = True)
                if f"nb_unique_pairwise_distance_influenced{ending}" in topo_log and phase == "train":
                    self.log(f"unique_pairwise_distance_influenced_nb{ending}_topo_analytics",topo_log[f"nb_unique_pairwise_distance_influenced{ending}"],on_step=False,on_epoch= True,prog_bar= False,logger = True)
                if f"topo_loss{ending}" in topo_log and phase == "train":
                    self.log(f"topo_loss{ending}_topo_analytics",topo_log[f"topo_loss{ending}"],on_step=False,on_epoch= True,prog_bar= False,logger = True)
                if f"rate_of_scale_calculation{ending}" in topo_log and phase == "train":
                    self.log(f"rate_of_topo_calculation{ending}_topo_analytics",topo_log[f"rate_of_scale_calculation{ending}"],on_step=False,on_epoch= True,prog_bar= False,logger = True)
        if "LR" in step_loss and phase == "train":
            self.log(f"LR",step_loss["LR"],on_step=False,on_epoch= True,prog_bar= False,logger = True)
        if "topo_weight_loss_epoch" in step_loss and phase == "train":
            self.log("topo_weight_loss_epoch",step_loss["topo_weight_loss_epoch"],on_step=True,on_epoch= False,prog_bar= False,logger = True)
        self.logger.log_metrics({f"{phase}_label_int":step_loss["label"]})
        self.logger.log_metrics({f"{phase}_pred_int":step_loss["prediction_int"]})


       
    def count_corrects(self, outputs):
        corrects = 0
        for output in outputs:
            if output["correct"]:
                corrects += 1
        return corrects, corrects / len(outputs)

    def validation_step(self, batch, batch_idx):
        bag, bag_label, dist_mat = batch
        model_output = self.model(bag)
        val_step_joint_loss = self.model.mil_loss_function(
            model_output[0], bag_label[0]
        )
        val_step_topo_loss = self.model.topo_loss_function(
            latent_distance_matrix=model_output[-1], input_distance_matrix=dist_mat
        )
        val_step_joint_loss.update(val_step_topo_loss)
        val_step_joint_loss["topo_weight_loss_epoch"] = self.topo_scheduler(val_step_joint_loss)
        
        if self.kill_mil_loss:
            val_step_joint_loss["loss"] = val_step_joint_loss["topo_weight_loss_epoch"] * val_step_joint_loss["topo_loss"]
        else:
            val_step_joint_loss["loss"] = (
                val_step_joint_loss["mil_loss"] + val_step_joint_loss["topo_weight_loss_epoch"] * val_step_joint_loss["topo_loss"]
            )
        self.val_confusion_matrix[
            val_step_joint_loss["label"], val_step_joint_loss["prediction_int"]
        ] += int(1)
        if self.class_weighting_factor:
            val_step_joint_loss["loss"] = val_step_joint_loss["loss"] * self.class_weights[val_step_joint_loss["label"]]
        self.log_step(val_step_joint_loss, "val")
        return val_step_joint_loss
    
    def test_step(self, batch, batch_idx):
        bag, bag_label = batch # distance matrix is not calculated for the test dataset
        model_output = self.model(bag)
        test_step_output = self.model.mil_loss_function(model_output[0], bag_label[0])
        self.test_confusion_matrix[
            test_step_output["label"], test_step_output["prediction_int"]
        ] += int(1)
        self.test_metrics.append(SCEMILA_Experiment.move_dict_tensors_to_cpu(test_step_output))
        return test_step_output
    
    def on_validation_epoch_end(self) -> None:
        self.log_confusion_matrix("val",self.val_confusion_matrix)
        self.val_confusion_matrix = torch.zeros(self.n_c, self.n_c)
        
    def on_train_epoch_end(self) -> None:
        p = 0.95
        self.log_confusion_matrix("train",self.train_confusion_matrix)
        #is_diagonal = torch.all(self.train_confusion_matrix == torch.diag(torch.diag(self.train_confusion_matrix)))
        # is_p_diagonal = (torch.sum(torch.diag(self.train_confusion_matrix)))/torch.sum(self.train_confusion_matrix) > p
        # if is_p_diagonal:
        #     self.model.remove_smoothing()
        current_smoothing = self.label_smoothing_scheduler.get_current_smoothing()
        self.model.set_mil_smoothing(current_smoothing)
        self.log("train_label_smoothing_epoch",current_smoothing,on_epoch=True)
        self.train_confusion_matrix = torch.zeros(self.n_c, self.n_c)

    def on_test_epoch_end(
            self 
        ):
        metrics_dict = {}
        self.log_confusion_matrix("test",self.test_confusion_matrix)
        self.test_confusion_matrix = torch.zeros(self.n_c, self.n_c)
                
        pred_int = torch.tensor([p["prediction_int"] for p in self.test_metrics])
        labels = torch.tensor([l["label"] for l in self.test_metrics])
        predictions = torch.cat([p["prediction"] for p in self.test_metrics])
        
        metrics_dict['accuracy'] = tf.accuracy(pred_int, labels, task="multiclass", num_classes=self.n_c, top_k=1,average= "micro").cpu().numpy().item()

        metrics_dict['precision_macro'] = tf.precision(pred_int, labels, task="multiclass", num_classes=self.n_c,average= "macro").cpu().numpy().item()
        
        metrics_dict['f1_macro'] = tf.f1_score(pred_int, labels, task='multiclass', num_classes=self.n_c, average='macro', top_k=1).cpu().numpy().item()
    
        metrics_dict['recall_macro'] = tf.recall(pred_int,labels, task='multiclass', num_classes=self.n_c, top_k=1, average="macro").cpu().numpy().item()
        
        metrics_dict['auroc'] = tf.auroc(predictions, labels, task='multiclass', num_classes=self.n_c)
        for key, value in metrics_dict.items():
            wandb.run.summary[key] = value
        self.log_dict(metrics_dict)  # this is what creates the table in the console, i guess the on_test_end hook is doing this
        try:
            if self.dataset is not None:
                from results.model_visualisation.instance_bag_SCEMILA_visulaizer import get_bag_and_instance_level_2D_embeddings
                get_bag_and_instance_level_2D_embeddings(model= self.model,dataset=self.dataset)
        except Exception as e:
            print("failed latent viz:")
            print(e)

    def predict_step(self, batch, batch_idx):
        assert False
        try:
            imgs, bag_label = batch
        except ValueError:
            imgs, bag_label, dist_matrix = batch
        imgs = imgs.squeeze(dim=0)
        self.curr_device = imgs.device
        result = self.model(imgs)
        return result

    def configure_optimizers(self):
        """optimizers. By default it is Adam. The options include RAdam, Ranger, and Adam_WU."""
        optimizer_class = get_module(
            self.params["optimizer"]["name"], self.params["optimizer"]["config"]
        )
        optimizer = optimizer_class(self.model)
        return optimizer

    def append_phase_to_dict(self, dict, phase):
        return {phase + "_" + key: value for key, value in dict.items()}
