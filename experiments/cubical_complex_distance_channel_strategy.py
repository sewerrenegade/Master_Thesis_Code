import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np



sys.path.append("/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code")

from results.metrics_descriptor import MetricsDescriptor
from distance_functions.distance_function_metrics.distance_matrix_metrics import (
    DistanceMatrixMetricCalculator,
)
from distance_functions.functions.basic_distance_functions import EuclideanDistance
from distance_functions.functions.cubical_complex_distance import (
    CubicalComplexImageDistanceFunction,
)
from configs.global_config import GlobalConfig
from datasets.image_augmentor import AugmentationSettings
import umap
import phate
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
import copy

# from datasets.SCEMILA.base_image_SCEMILA import SCEMILAimage_base,SCEMILA_fnl34_feature_base,SCEMILA_DinoBloom_feature_base
from datasets.SCEMILA import *
from datasets.dataset_factory import BASE_MODULES as DATA_SET_MODULES

DEFAULT_TRANSFROM_DICT = {
    "PHATE": (phate.PHATE, {"n_components": 3, "knn": 10, "decay": 40, "t": "auto"}),
    "TSNE": (TSNE, {"n_components": 3, "method": "exact"}),
    "Isomap": (Isomap, {"n_components": 3}),
    "UMAP": (umap.UMAP, {"n_components": 3}),
    "PCA": (PCA, {"n_components": 3}),
}

SWEEP_PORJECTION_DIM = GlobalConfig.DOWNPROJECTION_TEST_DIMENSIONS

DATASET_NAMES_AND_SETTINGS = {
    ("CIFAR10", "normal"): {
        "training_mode": True,
        "balance_dataset_classes": 100,
        "gpu": False,
        "augmentation_settings": AugmentationSettings(),
        "flatten": True,
        "numpy": True,
    },
    ("Acevedo", "normal"): {
        "training_mode": True,
        "balance_dataset_classes": 100,
        "gpu": False,
        "augmentation_settings": AugmentationSettings(),
        "flatten": True,
        "numpy": True,
        "resize": True
    },
}

descriptors = []


def perform_test():
    per_class_samples_for_metric_calc = 10
    metric = DistanceMatrixMetricCalculator
    experiment_metrics = {}

    for dataset_name, db_settings in DATASET_NAMES_AND_SETTINGS.items():
        experiment_metrics[dataset_name[0]] = {}
        iter_metric = experiment_metrics[dataset_name[0]]
        dataset_class = DATA_SET_MODULES.get(dataset_name[0])
        assert dataset_class is not None
        dataset = dataset_class(**db_settings)

        # Euclidean Baseline metrics
        baseline_metric_desc = MetricsDescriptor(
            metric_calculator=metric,
            dataset=dataset,
            distance_function=EuclideanDistance(),
            per_class_samples=per_class_samples_for_metric_calc,
        )
        baseline_metrics = baseline_metric_desc.calculate_metric()
        iter_metric["baseline"] = baseline_metrics
        # Normal RGB Cubical Complex metrics
        db_settings_cub = copy.deepcopy(db_settings)
        db_settings_cub["flatten"] = False
        dataset_cub = dataset_class(**db_settings_cub)
        cub_complex_metrics_desc = MetricsDescriptor(
            metric_calculator=metric,
            dataset=dataset_cub,
            distance_function=CubicalComplexImageDistanceFunction(),
            per_class_samples=int(per_class_samples_for_metric_calc / 2),
        )
        cub_complex_metrics = cub_complex_metrics_desc.calculate_metric()
        iter_metric["normal_cub_comp"] = cub_complex_metrics
        # Merged Channel Cubical Complex metrics
        cub_complex_join_metrics_desc = MetricsDescriptor(
            metric_calculator=metric,
            dataset=dataset_cub,
            distance_function=CubicalComplexImageDistanceFunction(join_channels=True),
            per_class_samples=int(per_class_samples_for_metric_calc / 2),
        )
        joint_cub_complex_metrics = cub_complex_join_metrics_desc.calculate_metric()
        iter_metric["joint_channel_cub_comp"] = joint_cub_complex_metrics
        # Grayscale input
        db_settings_tmp = copy.deepcopy(db_settings)
        db_settings_tmp["flatten"] = False
        db_settings_tmp["grayscale"] = True
        dataset_cub_grayscale = dataset_class(**db_settings_tmp)
        grayscale_cub_complex_metrics_description = MetricsDescriptor(
            metric_calculator=metric,
            dataset=dataset_cub_grayscale,
            distance_function=CubicalComplexImageDistanceFunction(),
            per_class_samples=int(per_class_samples_for_metric_calc / 2),
        )
        grayscale_cubical_complex_metrics = grayscale_cub_complex_metrics_description.calculate_metric()
        iter_metric["grayscale_cub_comp"] = grayscale_cubical_complex_metrics
    return experiment_metrics


def create_table_and_export_to_pdf(metrics_dict, key_metric='accuracy'):
    # Convert the metrics dictionary to a Pandas DataFrame
    rows = []

    def make_serializable(metrics):
        if isinstance(metrics, np.ndarray) and metrics.size == 1:
            return metrics.item()
        return metrics

    for dataset, metrics in metrics_dict.items():
        for metric_type, values in metrics.items():
            values = make_serializable(values)
            row = {'Dataset': dataset, 'Calculation': metric_type}
            metric_value = values.get(key_metric, float('-inf'))
            row.update({key_metric:metric_value})
            rows.append(row)

    df = pd.DataFrame(rows)

    # Export DataFrame to a PDF
    num_rows, num_cols = df.shape
    fig_width = num_cols * 2
    fig_height = num_rows * 0.3

    with PdfPages('metrics_results.pdf') as pdf:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
if __name__ == "__main__":
    exp_metrics = perform_test()
    create_table_and_export_to_pdf(exp_metrics,'knn_acc')
    
