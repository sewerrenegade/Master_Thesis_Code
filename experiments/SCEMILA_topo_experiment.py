import typing
import pytorch_lightning as pl
import torch
import torchmetrics.functional as tf
import wandb
from datasets.SCEMILA.SEMILA_indexer import SCEMILA_Indexer
from experiments.SCEMILA_experiment import DataMatrix, SCEMILA_Experiment
from models.SCEMILA.topo_SCEMILA_model import TopoAMiL
from models.model_factory import get_module
import matplotlib.pyplot as plt
import seaborn as sns


class TopoSCEMILA_Experiment(pl.LightningModule):
    def __init__(self, model: TopoAMiL, params: typing.Dict[str, typing.Any]) -> None:
        super(TopoSCEMILA_Experiment, self).__init__()
        self.model = model
        self.params = params
        self.n_c = self.params.get("num_class",5)
        self.class_weighting = self.params.get("class_weighting",False)
        self.curr_device = None
        self.current_data_object = DataMatrix()
        
        self.train_confusion_matrix,self.val_confusion_matrix,self.test_confusion_matrix = torch.zeros(self.n_c, self.n_c),torch.zeros(self.n_c, self.n_c),torch.zeros(self.n_c, self.n_c),
        self.test_metrics = []
        self.indexer = SCEMILA_Indexer.get_indexer()
        if self.class_weighting:
            self.class_weights = self.get_class_weights()
        pass
    
    def get_class_weights(self):
        train_class_distribution = self.indexer.class_sample_counts_in_train_set
        assert len(train_class_distribution) == self.n_c
        total_number = sum([sample_count for class_name,sample_count in train_class_distribution.items()])
        class_weights = {self.indexer.convert_from_int_to_label_bag_level(class_key):torch.tensor(total_number/(5*class_count)) for class_key,class_count in train_class_distribution.items()}
        return class_weights
        

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)

    def convert_matrix_to_str(self, mat):
        return str(mat.tolist())
    
    def log_confusion_matrix(self,phase,confusion_matrix):
        plt.figure(figsize=(10, 7))
        labels = [self.indexer.convert_from_int_to_label_bag_level(i) for i in range(self.model.class_count)]
        sns.heatmap(confusion_matrix.cpu().numpy(), annot=True, fmt="g", cmap="Blues",xticklabels = labels,yticklabels=labels,cbar=False)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        wandb.log({f"{phase} confusion matrix": wandb.Image(plt)})
        plt.close()

    def training_step(self, batch, batch_idx):
        bag, bag_label, dist_mat = batch
        model_output = self.model(bag)
        train_joint_loss = self.model.mil_loss_function(model_output[0], bag_label[0])
        train_step_topo_loss = self.model.topo_loss_function(
            latent_distance_matrix=model_output[-1], input_distance_matrix=dist_mat
        )
        train_joint_loss.update(train_step_topo_loss)
        train_joint_loss["topo_loss_weight"] = self.model.lam_topo * (self.model.lam_topo_per_epoch_decay**self.current_epoch)
        train_joint_loss["loss"] = train_joint_loss["mil_loss"] + train_joint_loss["topo_loss_weight"] * train_joint_loss["topo_loss"]
        self.train_confusion_matrix[
            train_joint_loss["label"], train_joint_loss["prediction_int"]
        ] += int(1)
        train_joint_loss["LR"] = self.trainer.optimizers[0].param_groups[0]['lr']
        if self.class_weighting:
            train_joint_loss["loss"] = train_joint_loss["loss"] * self.class_weights[train_joint_loss["label"]]
        self.log_step(train_joint_loss, "train", progress_bar= True)
        return train_joint_loss
            
    def log_step(self,step_loss,phase,progress_bar = False):
        self.log(f"{phase}_loss",step_loss["loss"].data,on_step=True,on_epoch= True,prog_bar= progress_bar,logger = True)
        self.log(f"{phase}_correct",step_loss["correct"],on_step=True,on_epoch= True,prog_bar= progress_bar,logger = True)
        self.log(f"{phase}_mil_loss",step_loss["mil_loss"].data,on_step=True,on_epoch= True,prog_bar= progress_bar,logger = True)
        self.log(f"{phase}_topo_loss",step_loss["topo_loss"].data,on_step=True,on_epoch= True,prog_bar= progress_bar,logger = True)
        if "LR" in step_loss and phase == "train":
            self.log(f"LR",step_loss["LR"],on_step=False,on_epoch= True,prog_bar= False,logger = True)
        if "topo_loss_weight" in step_loss and phase == "train":
            self.log("topo_loss_weight",step_loss["topo_loss_weight"],on_step=False,on_epoch= True,prog_bar= False,logger = True)
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
        val_step_joint_loss["topo_loss_weight"] = min(self.model.lam_topo * (self.model.lam_topo_per_epoch_decay**self.current_epoch),self.model.max_lam)
        val_step_joint_loss["loss"] = (
            val_step_joint_loss["mil_loss"] + val_step_joint_loss["topo_loss_weight"] * val_step_joint_loss["topo_loss"]
        )
        self.val_confusion_matrix[
            val_step_joint_loss["label"], val_step_joint_loss["prediction_int"]
        ] += int(1)
        if self.class_weighting:
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
        self.log_confusion_matrix("train",self.train_confusion_matrix)
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
