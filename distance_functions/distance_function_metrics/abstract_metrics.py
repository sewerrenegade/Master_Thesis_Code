from abc import ABC, abstractmethod
import numpy as np
class Metric(ABC):
    @abstractmethod
    def get_settings():
        pass
    @abstractmethod
    def calculate_metric():
        pass
    
    @staticmethod
    def get_all_scalar_metrics(metrics):
        scalar_metrics_list = [{k: v for k, v in d.items() if k != "metric"} for d in metrics]
        for indx,metric in enumerate(metrics):
            scalar_metrics_list[indx].update({key:value.item()  for key,value in metric["metric"].items() if value.size == 1 and np.issubdtype(value.dtype, np.number)})
        return scalar_metrics_list