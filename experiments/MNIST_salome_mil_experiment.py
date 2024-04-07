import typing
import pytorch_lightning as pl
import torch
import torchmetrics.functional as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score

from models.model_factory import get_module
from models.salome_models.mil import CV_MIL
from models.salome_models.topo_reg_mil import CV_TopoRegMIL


class MILEXperiment_CV(pl.LightningModule):
    def __init__(
        self, 
        model: CV_MIL, 
        params: typing.Dict[str, typing.Any]
    ) -> None:
        super(MILEXperiment_CV, self).__init__()
        self.model = model
        self.params = params
        self.n_c = self.params["num_class"]
        self.curr_device = None
        self.hold_graph = False

        try:
            self.hold_graph = self.params["retain_first_backpass"]
        except KeyError:
            pass

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)

    def training_step(
            self, 
            batch, 
            batch_idx
        ):
        imgs, bag_label = batch
        imgs = imgs.squeeze(dim=0)
        self.curr_device = imgs.device
        result = self.model(imgs, bag_label)
        train_loss = self.model.mil_loss_function0(result, is_training=True, epoch_num=self.current_epoch)
        self.log_dict({f"train_{k}": v for k, v in train_loss.items()}, on_step=True, on_epoch=True)
        return train_loss

    def validation_step(
            self, 
            batch, 
            batch_idx
        ):
        imgs, bag_label = batch
        imgs = imgs.squeeze(dim=0)
        self.curr_device = imgs.device
        result = self.model(imgs, bag_label)
        pred = torch.argmax(result.bag_prediction)
        labels_oh = torch.zeros(result.bag_prediction.size())
        labels_oh[0, bag_label] = 1
        val_loss = self.model.mil_loss_function0(result, is_training=False, epoch_num=self.current_epoch)
        self.log_dict({f"val_{k}": v for k, v in val_loss.items()})
        validation_output = {"val_loss": val_loss['loss'], "labels": result.bag_label.item(), "preds_int": pred, "preds": result.bag_prediction}
        return validation_output
    
    def validation_epoch_end(self, outputs):
        pred_int = torch.tensor([p["preds_int"] for p in outputs])
        labels = torch.tensor([l["labels"] for l in outputs])
        f1 = tf.f1_score(pred_int, labels, task='multiclass', num_classes=self.n_c, average='weighted', top_k=1)
        self.log("val_f1_macro", f1)

    def test_step(
            self, 
            batch, 
            batch_idx
        ):
        imgs, bag_label = batch
        imgs = imgs.squeeze(dim=0)
        self.curr_device = imgs.device
        result = self.model(imgs, bag_label)
        pred = torch.argmax(result.bag_prediction)
        labels_oh = torch.zeros(result.bag_prediction.size())
        labels_oh[0, bag_label] = 1
        test_output = {"labels": result.bag_label, "preds_int": pred, "preds": result.bag_prediction, "labels_oh": labels_oh}
        return test_output
    
    def test_epoch_end(
            self, 
            outputs, 
        ):
        pred_int = torch.tensor([p["preds_int"] for p in outputs])
        labels = torch.tensor([l["labels"] for l in outputs])
        predictions = torch.cat([p["preds"] for p in outputs])
        lebels_oh = torch.cat([l["labels_oh"] for l in outputs])

        # recall = calculate_recall(pred_int, labels, num_classes=self.n_c)
        # precision = calculate_precision(pred_int, labels, num_classes=self.n_c)
        # sensitivity = calculate_sensitivity(pred_int, labels, num_classes=self.n_c)
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
        self.logger.experiment.add_figure("Confusion matrix", fig_)

        result_dict = {
            "Accuracy": acc,
            "F1_macro_weighted": f1,
            "AUROC": auroc,
            "Prroc": prroc,
            "Recall": recall,
            "Precision": precision,
            # "Sensitivity": sensitivity,
            }

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
    

class TopoRegMILEXperiment_CV(pl.LightningModule):
    def __init__(
        self, 
        model: CV_TopoRegMIL, 
        params: typing.Dict[str, typing.Any]
    ) -> None:
        super(TopoRegMILEXperiment_CV, self).__init__()
        self.model = model
        self.params = params
        self.n_c = self.params["num_class"]
        self.curr_device = None
        self.hold_graph = False

        try:
            self.hold_graph = self.params["retain_first_backpass"]
        except KeyError:
            pass

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)

    def training_step(
            self, 
            batch, 
            batch_idx
        ):
        imgs, bag_label = batch
        imgs = imgs.squeeze(dim=0)
        self.curr_device = imgs.device
        result = self.model(imgs, bag_label)
        train_loss = self.model.loss_function(result, is_training=True, epoch_num=self.current_epoch)
        self.log_dict({f"train_{k}": v for k, v in train_loss.items()}, on_step=True, on_epoch=True)
        return train_loss

    def validation_step(
            self, 
            batch, 
            batch_idx
        ):
        imgs, bag_label = batch
        imgs = imgs.squeeze(dim=0)
        self.curr_device = imgs.device
        result = self.model(imgs, bag_label)
        pred = torch.argmax(result.bag_prediction)
        labels_oh = torch.zeros(result.bag_prediction.size())
        labels_oh[0, bag_label] = 1
        val_loss = self.model.loss_function(result, is_training=False, epoch_num=self.current_epoch)
        self.log_dict({f"val_{k}": v for k, v in val_loss.items()})
        validation_output = {"val_loss": val_loss['loss'], "labels": result.bag_label.item(), "preds_int": pred, "preds": result.bag_prediction}
        return validation_output
    
    def validation_epoch_end(self, outputs):
        pred_int = torch.tensor([p["preds_int"] for p in outputs])
        labels = torch.tensor([l["labels"] for l in outputs])

        f1 = tf.f1_score(pred_int, labels, task='multiclass', num_classes=self.n_c, average='weighted', top_k=1)

        self.log("val_f1_macro", f1)

    def test_step(
            self, 
            batch, 
            batch_idx
        ):
        imgs, bag_label = batch
        imgs = imgs.squeeze(dim=0)
        self.curr_device = imgs.device
        result = self.model(imgs, bag_label)
        pred = torch.argmax(result.bag_prediction)
        labels_oh = torch.zeros(result.bag_prediction.size())
        labels_oh[0, bag_label] = 1
        test_output = {"labels": result.bag_label, "preds_int": pred, "preds": result.bag_prediction, "labels_oh": labels_oh}
        return test_output
    
    def test_epoch_end(
            self, 
            outputs, 
        ):
        pred_int = torch.tensor([p["preds_int"] for p in outputs])
        labels = torch.tensor([l["labels"] for l in outputs])
        predictions = torch.cat([p["preds"] for p in outputs])
        lebels_oh = torch.cat([l["labels_oh"] for l in outputs])

        # recall = calculate_recall(pred_int, labels, num_classes=self.n_c)
        # precision = calculate_precision(pred_int, labels, num_classes=self.n_c)
        # sensitivity = calculate_sensitivity(pred_int, labels, num_classes=self.n_c)
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
        self.logger.experiment.add_figure("Confusion matrix", fig_)

        result_dict = {
            "Accuracy": acc,
            "F1_macro_weighted": f1,
            "AUROC": auroc,
            "Prroc": prroc,
            "Recall": recall,
            "Precision": precision,
            # "Sensitivity": sensitivity,
            }

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