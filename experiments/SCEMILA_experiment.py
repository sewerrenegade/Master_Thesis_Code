import typing
import pytorch_lightning as pl
import torch
import torchmetrics.functional as tf
import wandb
from datasets.SCEMILA.SEMILA_indexer import SCEMILA_Indexer
from models.model_factory import get_module
from models.SCEMILA.SCEMILA_model import AMiL
import matplotlib.pyplot as plt
import seaborn as sns


class SCEMILA_Experiment(pl.LightningModule):
    def __init__(
        self, 
        model: AMiL, 
        params: typing.Dict[str, typing.Any]
    ) -> None:
        super(SCEMILA_Experiment, self).__init__()
        self.model = model
        self.params = params
        self.n_c = self.params.get("num_class",5)
        self.class_weighting = self.params.get("class_weighting_factor",0.0)
        self.curr_device = None
        self.current_data_object =  DataMatrix()        
        self.train_confusion_matrix,self.val_confusion_matrix ,self.test_confusion_matrix  = torch.zeros(self.n_c, self.n_c),torch.zeros(self.n_c, self.n_c),torch.zeros(self.n_c, self.n_c)
        self.test_metrics = []
        self.indexer = SCEMILA_Indexer.get_indexer()
        self.class_weights = self.get_class_weights(self.class_weighting)
    
    def get_class_weights(self,weighting_factor):
        assert 0 <= weighting_factor <= 1
        train_class_distribution = self.indexer.class_sample_counts_in_train_set
        assert len(train_class_distribution) == self.n_c
        total_number = sum([sample_count for class_name,sample_count in train_class_distribution.items()])
        class_weights = {self.indexer.convert_from_int_to_label_bag_level(class_key):torch.tensor((1 - weighting_factor)+ weighting_factor*total_number/(self.n_c*class_count)) for class_key,class_count in train_class_distribution.items()}
        return class_weights
        
    
    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)
    
    def convert_matrix_to_str(self,mat):
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
    
    def training_step(
            self, 
            batch, 
            batch_idx
        ):
        bag, bag_label = batch
        model_output = self.model(bag)
        train_step_output = self.model.mil_loss_function(model_output[0],bag_label[0])
        self.train_confusion_matrix[train_step_output["label"], train_step_output["prediction_int"]] += 1
        train_step_output["LR"] = self.trainer.optimizers[0].param_groups[0]['lr']
        if self.class_weighting:
            train_step_output["loss"] = train_step_output["loss"] * self.class_weights[train_step_output["label"]]
        self.log_step(train_step_output,"train",progress_bar= True)
        return train_step_output
    
    def validation_step(
            self, 
            batch, 
            batch_idx
        ):
        bag, bag_label = batch
        model_output  = self.model(bag)
        validation_step_output = self.model.mil_loss_function(model_output[0],bag_label[0])
        self.val_confusion_matrix[validation_step_output["label"], validation_step_output["prediction_int"]] += 1
        if self.class_weighting:
            validation_step_output["loss"] = validation_step_output["loss"] * self.class_weights[validation_step_output["label"]]
        self.log_step(validation_step_output,"val")
        return validation_step_output
    
    def log_step(self,step_loss,phase,progress_bar = False):
        self.log(f"{phase}_loss",step_loss["loss"].data,on_step=True,on_epoch= True,prog_bar= progress_bar,logger = True)
        self.log(f"{phase}_mil_loss",step_loss["loss"].data,on_step=True,on_epoch= True,prog_bar= False,logger = True)
        self.log(f"{phase}_correct",step_loss["correct"],on_step=True,on_epoch= True,prog_bar= progress_bar,logger = True)
        if "LR" in step_loss and phase == "train":
            self.log(f"LR",step_loss["LR"],on_step=False,on_epoch= True,prog_bar= False,logger = True)
        self.logger.log_metrics({f"{phase}_label_int":step_loss["label"]})
        self.logger.log_metrics({f"{phase}_pred_int":step_loss["prediction_int"]})
       
    def on_validation_epoch_end(self) -> None:
        self.log_confusion_matrix("val",self.val_confusion_matrix)
        self.val_confusion_matrix = torch.zeros(self.n_c, self.n_c)
    
    def on_train_epoch_end(self) -> None:
        self.log_confusion_matrix("train",self.train_confusion_matrix)
        is_diagonal = torch.all(self.train_confusion_matrix == torch.diag(torch.diag(self.train_confusion_matrix)))
        if is_diagonal:
            self.model.remove_smoothing()
        self.train_confusion_matrix = torch.zeros(self.n_c, self.n_c)


    def count_corrects(self,outputs):
        corrects = 0
        for output in outputs:
            if output["correct"]:
                corrects+=1
        return corrects,corrects/len(outputs)

    def test_step(
            self, 
            batch, 
            batch_idx
        ):
        bag, bag_label = batch
        model_output  = self.model(bag)
        test_step_output = self.model.mil_loss_function(model_output[0],bag_label[0])
        #self.log_step(test_step_output,model_output,batch,"test")
        self.test_confusion_matrix[test_step_output["label"], test_step_output["prediction_int"]] += int(1)
        self.test_metrics.append(SCEMILA_Experiment.move_dict_tensors_to_cpu(test_step_output))

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

    def predict_step(
            self, 
            batch, 
            batch_idx
        ):
        imgs, bag_label = batch
        imgs = imgs.squeeze(dim=0)
        self.curr_device = imgs.device
        result = self.model(imgs)
        return result

    def configure_optimizers(self):
        """ optimizers. By default it is Adam. The options include RAdam, Ranger, and Adam_WU. """
        optimizer_class = get_module(
            self.params["optimizer"]["name"], self.params["optimizer"]["config"]
        )
        optimizer = optimizer_class(self.model)
        return optimizer
    
    def append_phase_to_dict(self,dict,phase):
        return {phase + '_' + key: value for key, value in dict.items()}
    
    @staticmethod
    def move_dict_tensors_to_cpu(data_dict):
        """
        Takes a dictionary as input and returns a new dictionary with all torch tensors moved to the CPU.
        
        Args:
            data_dict (dict): A dictionary potentially containing torch tensors on the GPU.
        
        Returns:
            dict: A new dictionary with all torch tensors moved to the CPU.
        """
        cpu_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                cpu_dict[key] = value.cpu()
            else:
                cpu_dict[key] = value
        return cpu_dict
        


class DataMatrix():
    '''DataMatrix contains all information about patient classification for later storage.
    Data is stored within a dictionary:

    self.data_dict[true entity] contains another dictionary with all patient paths for
                                the patients of one entity (e.g. AML-PML-RARA, SCD, ...)

    --> In this dictionary, the paths form the keys to all the data of that patient
        and it's classification, stored as a tuple:

        - attention_raw:    attention for all single cell images before softmax transform
        - attention:        attention after softmax transform
        - prediction:       Numeric position of predicted label in the prediction vector
        - prediction_vector:Prediction vector containing the softmax-transformed activations
                            of the last AMiL layer
        - loss:             Loss for that patients' classification
        - out_features:     Aggregated bag feature vectors after attention calculation and
                            softmax transform. '''

    def __init__(self):
        self.data_dict = dict()

    def add_patient(
            self,
            entity,
            path_full,
            attention_raw,
            attention,
            prediction,
            prediction_vector,
            loss,
            out_features):
        '''Add a new patient into the data dictionary. Enter all the data packed into a tuple into the dictionary as:
        self.data_dict[entity][path_full] = (attention_raw, attention, prediction, prediction_vector, loss, out_features)

        accepts:
        - entity: true patient label
        - path_full: path to patient folder
        - attention_raw: attention before softmax transform
        - attention: attention after softmax transform
        - prediction: numeric bag label
        - prediction_vector: output activations of AMiL model
        - loss: loss calculated from output actiations
        - out_features: bag features after attention calculation and matrix multiplication

        returns: Nothing
        '''

        if not (entity in self.data_dict):
            self.data_dict[entity] = dict()
        self.data_dict[entity][path_full] = (
            attention_raw.detach().cpu().numpy(),
            attention.detach().cpu().numpy(),
            prediction,
            prediction_vector.data.cpu().numpy()[0],
            float(
                loss.data.cpu()),
            out_features.detach().cpu().numpy())

    def return_data(self):
        return self.data_dict
