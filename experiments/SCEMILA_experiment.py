import typing
import pytorch_lightning as pl
import torch
import copy
import torchmetrics.functional as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score
import torch.nn.functional as F

from models.model_factory import get_module
from models.SCEMILA.model import AMiL
from experiments.metrics import calculate_recall,calculate_precision, calculate_sensitivity
from models.salome_models.topo_reg_mil import CV_TopoRegMIL


class SCEMILA_Experiment(pl.LightningModule):
    def __init__(
        self, 
        model: AMiL, 
        params: typing.Dict[str, typing.Any]
    ) -> None:
        super(SCEMILA_Experiment, self).__init__()
        self.model = model
        self.params = params
        self.n_c = self.params["num_class"]
        self.curr_device = None
        self.hold_graph = False
        self.best_loss = 10
        self.no_improvement_for = 0
        self.current_data_object =  DataMatrix()        
        self.current_confusion_matrix = torch.zeros(self.n_c, self.n_c)
        self.best_model = copy.deepcopy(self.model.state_dict())
        try:
            self.hold_graph = self.params["retain_first_backpass"]
        except KeyError:
            pass

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)
    
    def convert_matrix_to_str(self,mat):
        return str(mat.tolist())
    def convert_matrix_to_pic(self,mat,name = "Confusion Matrix"):
        return [pl.Image(mat, caption=name)]
    
    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_closure,
    # ):
    #     optimizer.step(closure=optimizer_closure)


    def training_step(
            self, 
            batch, 
            batch_idx
        ):
        bag, bag_label,path = batch
        prediction, att_raw, att_softmax, bag_feature_stack  = self.model(batch)
        train_step_output = self.model.mil_loss_function(prediction,bag_label[0])
        label_prediction = torch.argmax(prediction, dim=1).item()
        label_groundtruth = bag_label[0].item()
        self.current_confusion_matrix[label_groundtruth, label_prediction] += int(1)
        #self.log_dict({f"train_{train_loss}"}, on_step=True, on_epoch=True)
        train_step_output["bag_size"] = len(bag)
        self.current_data_object.add_patient(
                label_groundtruth,
                path[0],
                att_raw,
                att_softmax,
                label_prediction,
                F.softmax(
                    prediction,
                    dim=1),
                train_step_output["train_loss"],
                bag_feature_stack)
        return train_step_output
    def on_epoch_end_custom(self):
        self.current_data_object =  DataMatrix()        
        self.current_confusion_matrix = torch.zeros(self.n_c, self.n_c)

    def training_epoch_end(self, outputs):
        total_loss,per_sample_avg_loss = self.sum_loss(outputs)
        corrects,accuracy = self.count_corrects(outputs)
        train_loss = per_sample_avg_loss
        self.log_dict({"train_loss":train_loss,"accuracy": accuracy, "confusion_matrix":self.convert_matrix_to_str(self.current_confusion_matrix),"data_obj":self.current_data_object.return_data()})
        self.on_epoch_end_custom()

    def count_corrects(self,outputs):
        corrects = 0
        for output in outputs:
            if output["correct"]:
                corrects+=1
        return corrects,corrects/len(outputs)

    def sum_loss(self,outputs):
        loss = 0.0
        for output in outputs:
            if output["train_loss"]:
                loss+= output["train_loss"]
        return loss,loss/len(outputs)


    def validation_step(
            self, 
            batch, 
            batch_idx
        ):
        bag, bag_label,path = batch
        prediction, att_raw, att_softmax, bag_feature_stack  = self.model(batch)
        validation_step_output = self.model.mil_loss_function(prediction,bag_label[0])
        label_prediction = torch.argmax(prediction, dim=1).item()
        label_groundtruth = bag_label[0].item()
        self.current_confusion_matrix[label_groundtruth, label_prediction] += int(1)
        #self.log_dict({f"train_{train_loss}"}, on_step=True, on_epoch=True)
        validation_step_output["bag_size"] = len(bag)
        self.current_data_object.add_patient(
                label_groundtruth,
                path[0],
                att_raw,
                att_softmax,
                label_prediction,
                F.softmax(
                    prediction,
                    dim=1),
                validation_step_output["train_loss"],
                bag_feature_stack)
        return validation_step_output
    
    def validation_epoch_end(self, outputs):
        total_loss,per_sample_avg_loss = self.sum_loss(outputs)
        corrects,accuracy = self.count_corrects(outputs)
        train_loss = per_sample_avg_loss
        self.log_dict({"train_loss":train_loss,"accuracy": accuracy, "confusion_matrix":self.convert_matrix_to_str(self.current_confusion_matrix),"data_obj":self.current_data_object.return_data()})

        pred_int = torch.tensor([p["prediction"] for p in outputs])
        labels = torch.tensor([l["label"] for l in outputs])
        f1 = tf.f1_score(pred_int, labels, task='multiclass', num_classes=self.n_c, average='weighted', top_k=1)
        self.log("val_f1_macro", f1)
        self.on_epoch_end_custom()

    def test_step(
            self, 
            batch, 
            batch_idx
        ):
        bag, bag_label,path = batch
        prediction, att_raw, att_softmax, bag_feature_stack  = self.model(batch)
        test_step_output = self.model.mil_loss_function(prediction,bag_label[0])
        label_prediction = torch.argmax(prediction, dim=1).item()
        label_groundtruth = bag_label[0].item()
        self.current_confusion_matrix[label_groundtruth, label_prediction] += int(1)
        #self.log_dict({f"train_{train_loss}"}, on_step=True, on_epoch=True)
        test_step_output["bag_size"] = len(bag)
        self.current_data_object.add_patient(
                label_groundtruth,
                path[0],
                att_raw,
                att_softmax,
                label_prediction,
                F.softmax(
                    prediction,
                    dim=1),
                test_step_output["train_loss"],
                bag_feature_stack)
        return test_step_output
    
    def test_epoch_end(
            self, 
            outputs, 
        ):
        pred_int = torch.tensor([p["prediction_int"] for p in outputs])
        labels = torch.tensor([l["label"] for l in outputs])
        predictions = torch.cat([p["prediction"] for p in outputs])
        #lebels_oh = torch.cat([l["labels_oh"] for l in outputs])

        recall = calculate_recall(pred_int, labels, num_classes=self.n_c)
        precision = calculate_precision(pred_int, labels, num_classes=self.n_c)
        sensitivity = calculate_sensitivity(pred_int, labels, num_classes=self.n_c)
        recall = recall_score(y_true=labels, y_pred=pred_int, average='macro')
        precision = precision_score(y_true=labels, y_pred=pred_int, average='macro')

        acc = tf.accuracy(pred_int, labels, task="multiclass", num_classes=self.n_c, top_k=1)
        f1 = tf.f1_score(pred_int, labels, task='multiclass', num_classes=self.n_c, average='weighted', top_k=1)

        confmat = tf.confusion_matrix(pred_int, labels, task='multiclass', num_classes=self.n_c)
        auroc = tf.auroc(predictions.cpu(), labels.cpu(), task='multiclass', num_classes=self.n_c)
        prroc = tf.average_precision(predictions.cpu(), labels.cpu(), task="multiclass", num_classes=self.n_c)

        self.log("Accuracy", acc)
        self.log("F1_macro_weighted", f1)
        self.log("AUROC", auroc)
        self.log("Prroc", prroc)
        self.log("Recall", recall)
        self.log("Precision", precision)
        # self.log("Sensitivity", sensitivity)

        df_cm = pd.DataFrame(confmat.numpy(), index = range(self.n_c), columns=range(self.n_c))
        plt.figure(figsize = (self.n_c,self.n_c))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)

        self.log("Confusion matrix", plt)

        result_dict = {
            "Accuracy": acc,
            "F1_macro_weighted": f1,
            "AUROC": auroc,
            "Prroc": prroc,
            "Recall": recall,
            "Precision": precision,
            # "Sensitivity": sensitivity,
            }
        self.on_epoch_end_custom()

        return result_dict

    def predict_step(
            self, 
            batch, 
            batch_idx
        ):
        imgs, bag_label = batch
        imgs = imgs.squeeze(dim=0)
        self.curr_device = imgs.device
        result = self.model(imgs, bag_label)
        return result

    def configure_optimizers(self):
        """ optimizers. By default it is Adam. The options include RAdam, Ranger, and Adam_WU. """
        optimizer_class = get_module(
            self.params["optimizer"]["name"], self.params["optimizer"]["config"]
        )
        optimizer = optimizer_class(self.model)
        return optimizer
    


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
