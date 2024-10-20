from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from torch import sum as t_sum, diag as t_diag

class LabelSmoothingScheduler:
    def __init__(self,experiment:pl.LightningModule,smoothing:float = 0.1,per_epoch_decay:float = 1.0,train_correct_threshold:float = 0.95, hold_high = True,default_value = 0.0):
        self.smoothing = smoothing
        self.hold_high = hold_high
        self.per_epoch_delay = per_epoch_decay
        self.train_correct_threshold = train_correct_threshold
        self.experiment = experiment
        self.threshold_achieved = False
        self.default_value = default_value

    def get_current_smoothing(self):
        if (self.threshold_achieved and self.hold_high) or (self.experiment.current_epoch != 0 and (t_sum(t_diag(self.experiment.train_confusion_matrix)))/t_sum(self.experiment.train_confusion_matrix) >= self.train_correct_threshold):
            smoothing_now = self.default_value
            self.threshold_achieved = True
        else:
            smoothing_now = self.smoothing*self.per_epoch_delay**self.experiment.current_epoch
        return smoothing_now