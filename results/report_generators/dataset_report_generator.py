

import os
from reportlab.platypus import  Table, TableStyle, Paragraph, Image, PageBreak,Spacer
from typing import Tuple, Union
from configs.global_config import GlobalConfig
from datasets.dataset_descriptor import SerializableDatasetDescriptor
from datasets.image_augmentor import Augmentability, AugmentationSettings

from results.report_generators.common_report_data import HEADING_STYLES,NORMAL_STYLE
from results.report_generators.common_report_functions import get_histogram_elements, dict_list_to_dataframe, plot_histograms, sort_dict_list,save_tensor_as_png,get_sample_image_elements



class DatasetReportElements:
    def __init__(self) -> None:
        self.name = None
        self.multiple_instance_dataset = []
        self.bag_size = []
        self.number_of_channels= []
        self.output_dimension= []
        self.augmentation_scheme = []
        self.class_distribution= []
        self.classes= []
        self.augmentation_settings= []
        self.dino_bloom = []
        self.size = []
        self.dataset_sample_image_paths = []
        self.result_elements = []
        self.make_sure_temp_results_folder_exists()
    
    def make_sure_temp_results_folder_exists(self):
        if not os.path.exists(GlobalConfig.TEMP_RESULTS_FOLDER):
            os.makedirs(GlobalConfig.TEMP_RESULTS_FOLDER)

        
    def add_variant(self,dataset):
        serializable_dataset_descriptor = SerializableDatasetDescriptor(dataset=dataset)
        if self.name is None:
            self.name = serializable_dataset_descriptor.name
            self.classes = serializable_dataset_descriptor.classes
        assert self.name == serializable_dataset_descriptor.name
        assert self.classes == serializable_dataset_descriptor.classes
        
        self.multiple_instance_dataset.append(serializable_dataset_descriptor.multiple_instance_dataset)
        self.bag_size.append(serializable_dataset_descriptor.bag_size)
        self.number_of_channels.append(serializable_dataset_descriptor.number_of_channels)
        self.output_dimension.append(serializable_dataset_descriptor.output_dimension)
        self.augmentation_scheme.append(serializable_dataset_descriptor.augmentation_scheme)
        self.class_distribution.append(serializable_dataset_descriptor.class_distribution)
        self.dino_bloom.append(serializable_dataset_descriptor.dino_bloom)
        self.size.append(serializable_dataset_descriptor.size)
        self.dataset_sample_image_paths.append(self.get_sample_image_paths(dataset))
        self.augmentation_settings.append(serializable_dataset_descriptor.augmentation_settings)
        
    def get_sample_image_paths(self,dataset):     
        per_class_image_tensors = dataset.get_per_class_image_samples()
        per_class_image_paths = {}
        for class_name, tensor_images in per_class_image_tensors.items():
            per_class_image_paths[class_name] = save_tensor_as_png(tensor_images[0])
        return per_class_image_paths
    def get_sample_image_elements(self):
        elements = []
        for path_dict in self.dataset_sample_image_paths:
            elements.extend(get_sample_image_elements(path_dict))
        return elements
            
            
    def get_elements(self):
        elements  = []
        elements.append(Paragraph(f"{self.name} Dataset Experiment Overview", HEADING_STYLES[3]))
        elements.append(Paragraph(f"Name: {self.name}", NORMAL_STYLE))
        elements.append(Paragraph(f"Output Dimension: {self.output_dimension}", NORMAL_STYLE))
        elements.append(Paragraph(f"Dataset Size: {self.size}", NORMAL_STYLE))
        elements.append(Paragraph(f"Augmentation Settings: {[augmentation_set.__repr__() for augmentation_set in self.augmentation_settings]}", NORMAL_STYLE))
        elements.append(Paragraph(f"Augmentation Scheme: {[augmentation_scheme.to_string() for augmentation_scheme in self.augmentation_scheme]}", NORMAL_STYLE))
        elements.append(Paragraph(f"Classes: {self.classes}", NORMAL_STYLE))
        elements.append(Paragraph(f"Uses DinoBloom Encoding: {self.dino_bloom}", NORMAL_STYLE))
        elements.append(Paragraph(f"Number of Output Channels: {self.number_of_channels}", NORMAL_STYLE))
        elements.append(Paragraph(f"Is Multiple Instance Dataset: {self.multiple_instance_dataset}", NORMAL_STYLE))
        elements.append(Paragraph(f"Bag Sizes: {self.bag_size}", NORMAL_STYLE))
        elements.append(Paragraph(f"Class Distribution:", NORMAL_STYLE))
        elements.extend(self.get_histogram_elements())
        elements.append(Paragraph(f"Sample Images:", NORMAL_STYLE))
        elements.extend(self.get_sample_image_elements())
        elements.extend(self.result_elements)
        return elements
        
    def get_histogram_elements(self):
        sorted_dict_list = sort_dict_list(self.class_distribution)
        histograms = plot_histograms(sorted_dict_list)
        histogram_elements = get_histogram_elements(histograms, n_per_row=2)

        return histogram_elements
    
    
