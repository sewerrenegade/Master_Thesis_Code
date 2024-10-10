import typing

import torch_optimizer as torch_optim
from torch import optim


class RAdam:
    def __init__(
        self,
        LR: float,
        betas: typing.Tuple[float, float] = (0.9, 0.99999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        self.lr = LR
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def __call__(self, model):
        optimizer = torch_optim.RAdam(
            model.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        return optimizer


class Ranger:
    def __init__(
        self,
        LR: float = 5e-4,
        alpha: float = 0.1,
        k: int = 6,
        N_sma_threshhold: int = 3,
        betas: typing.Tuple[float, float] = (0.9, 0.99999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        self.lr = LR
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.k = k
        self.N_sma_threshhold = N_sma_threshhold

    def __call__(self, model):
        optimizer = torch_optim.Ranger(
            model.parameters(),
            lr=self.lr,
            alpha=self.alpha,
            k=self.k,
            N_sma_threshhold=self.N_sma_threshhold,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        return optimizer

class SGD:
    def __init__(
        self,
        LR: float,
        momentum = 0.9,
        nestrov = True
    ):
        self.lr = LR
        self.momentum = momentum
        self.nestrov = nestrov

    def __call__(self, model):
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum= self.momentum,
            nesterov= self.nestrov
        )
        return optimizer

class Adam:
    def __init__(
        self,
        LR,
        weight_decay=0,
        scheduler = None,
        monitor_metric = "val_correct_epoch",
        mode = "max",
        factor=0.5,
        patience=5
    ):
        self.lr = LR
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.monitor_metric = monitor_metric
        self.metric_mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = True

    def __call__(self, model):
        optims = []
        optims = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if self.scheduler is None:
            return optims
        else:
            if self.scheduler == "ReduceLROnPlateau":
                sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer= optims, mode=self.metric_mode,factor=self.factor,patience= self.patience, threshold=0.01, verbose= self.verbose,min_lr=0.0000001)
                scheduler = {
                    'scheduler': sched, 
                    'monitor': self.monitor_metric
                }
                return [optims], [scheduler]
            else:
                raise ValueError("This type of scheduler is not supported")