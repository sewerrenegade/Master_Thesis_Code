from datasets.base_dataset_abstraction import BaseDataset
from datasets.dataset_descriptor import SerializableDatasetDescriptor
from datasets.embedded_datasets.dataset.embedding_base import EmbeddingBaseDataset
from datasets.mil_dataset_abstraction import BaseMILDataset
from typing import Union,Type
from distance_functions.distance_function_metrics.abstract_metrics import Metric



class MetricsDescriptor:
    def __init__(self,metric_calculator: Type[Metric],
                dataset: Union[BaseDataset, BaseMILDataset, EmbeddingBaseDataset],
                distance_function: object,
                per_class_samples: int,
                metric_calculator_settings :dict = {},
            ):
        self.dataset= dataset
        self.distance_function=distance_function
        self.per_class_samples=per_class_samples
        self.metric_calculator= metric_calculator(self,**metric_calculator_settings)

    def calculate_metric(self,recalculate = False):
        from results.results_manager import ResultsManager
        res_manager = ResultsManager.get_manager()
        if not recalculate and res_manager.check_if_result_already_exists(self):
            metric = res_manager.load_metric(self)
        else:
            metric = self.metric_calculator.calculate_metric()
            res_manager.save_results(descriptor=self,results={"metrics_dict":metric})
            metric = res_manager.load_metric(self)
        return metric
        
    def to_dict(self):
        return self.create_serialializable_descriptor_from_live_metrics_descriptor().to_dict()
    
    def create_serialializable_descriptor_from_live_metrics_descriptor(
        self
        ):
        desc = self
        if isinstance(desc.dataset, EmbeddingBaseDataset):
            serializable_dataset_descriptor = (
                desc.dataset.get_serializable_embedding_descriptor()
            )
        elif isinstance(desc.dataset, Union[BaseDataset, BaseMILDataset]):
            serializable_dataset_descriptor = SerializableDatasetDescriptor(
                dataset=desc.dataset
            )
        metric_name = desc.metric_calculator.name
        metric_settings = desc.metric_calculator.get_settings()
        distance_function_name = desc.distance_function.name
        distance_function_settings = desc.distance_function.get_settings()
        per_clas_samples = desc.per_class_samples
        serializable_desc = SerializableMetricsDescriptor(
            serializable_dataset_descriptor=serializable_dataset_descriptor,
            metric_name=metric_name,
            metric_settings=metric_settings,
            distance_function_name=distance_function_name,
            distance_function_settings=distance_function_settings,
            per_class_samples=per_clas_samples,
        )
        return serializable_desc



class SerializableMetricsDescriptor:
    
    def __init__(
        self,
        serializable_dataset_descriptor,
        metric_name: str,
        metric_settings: dict,
        distance_function_name: str,
        distance_function_settings: dict,
        per_class_samples: int,
    ):
        from datasets.embedded_datasets.generators.embedding_descriptor import SerializableEmbeddingDescriptor
        assert isinstance(serializable_dataset_descriptor,(SerializableEmbeddingDescriptor, SerializableDatasetDescriptor))
        self.dataset_dict = serializable_dataset_descriptor
        self.metric_name = metric_name
        self.metric_settings = metric_settings
        self.distance_function_name = distance_function_name
        self.distance_function_settings = distance_function_settings
        self.per_class_samples = per_class_samples


    def to_dict(self):
        return {
            'dataset_dict': self.dataset_dict.to_dict() if hasattr(self.dataset_dict, 'to_dict') else self.dataset_dict,
            'metric_name': self.metric_name,
            'metric_settings': self.metric_settings,
            'distance_function_name': self.distance_function_name,
            'distance_function_settings': self.distance_function_settings,
            'per_class_samples': self.per_class_samples,
        }