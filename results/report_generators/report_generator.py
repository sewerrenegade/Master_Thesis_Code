import sys


sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
import pandas as pd

from results.report_generators.dataset_report_generator import DatasetReportElements
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, PageBreak
from reportlab.lib import colors
from results.metrics_descriptor import MetricsDescriptor
from datasets.embedded_datasets.dataset.embedding_base import EmbeddingBaseDataset
from distance_functions.distance_function_metrics.distance_matrix_metrics import DistanceMatrixMetricCalculator
from distance_functions.functions.basic_distance_functions import EuclideanDistance, L1Distance
from distance_functions.functions.cubical_complex_distance import CubicalComplexImageDistanceFunction
from results.results_manager import ResultsManager
from datasets.embedded_datasets.generators.embedding_descriptor import EmbeddingDescriptor
from configs.global_config import GlobalConfig
from datasets.image_augmentor import AugmentationSettings
import umap
import phate
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
import copy
from datasets.SCEMILA import *
from datasets.dataset_factory import BASE_MODULES as DATA_SET_MODULES

DEFAULT_TRANSFROM_DICT = {
    "PHATE": (phate.PHATE, {'n_components': 8, 'knn': 10, 'decay': 40, 't': 'auto'}),
    "TSNE": (TSNE, {'n_components': 8, 'method': 'exact'}),
    "Isomap": (Isomap, {'n_components': 8}),
    "UMAP": (umap.UMAP, {'n_components': 8}),
    "PCA": (PCA, {'n_components': 8})
}
METRIC = DistanceMatrixMetricCalculator
SWEEP_PORJECTION_DIM = GlobalConfig.DOWNPROJECTION_TEST_DIMENSIONS

DATASET_NAMES_AND_SETTINGS = {
    ("FashionMNIST"): {"training_mode": True, "balance_dataset_classes": 10, "gpu": False, "augmentation_settings": AugmentationSettings(), "flatten": False, "numpy": True},
    ("SCEMILA/image_data"): {"training_mode": True, "balance_dataset_classes": 10, "gpu": False, "augmentation_settings": AugmentationSettings(), "flatten": False, "numpy": True},
    ("Acevedo"): {"training_mode": True, "balance_dataset_classes": 10, "gpu": False, "augmentation_settings": AugmentationSettings(), "flatten": False, "numpy": True},
    
    ("CIFAR10"): {"training_mode": True, "balance_dataset_classes": 10, "gpu": False, "augmentation_settings": AugmentationSettings(), "flatten": False, "numpy": True},
    ("MNIST"): {"training_mode": True, "balance_dataset_classes": 10, "gpu": False, "augmentation_settings": AugmentationSettings(), "flatten": False, "numpy": True},
}
AUGMENTATIONS_OF_INTEREST = ['all', 'none']
descriptors = []

def create_complete_metrics_table(experiment_metrics, key_metric='triplet_loss'):
    combined_metrics = []
    for augmentation, datasets in experiment_metrics.items():
        for dataset_name, dataset_types in datasets.items():
            for calc_type, calc_results in dataset_types.items():
                metric_value = calc_results.get(key_metric, float('-inf'))
                entry = {
                    'Dataset': dataset_name,
                    'Calculation': calc_type,
                    'Augmentation': augmentation,
                    'KeyMetricValue': metric_value,
                }
                combined_metrics.append(entry)

    df = pd.DataFrame(combined_metrics)
    pivot_df = df.pivot_table(index=['Dataset', 'Calculation'], columns=['Augmentation'], values=['KeyMetricValue']).reset_index()
    pivot_df = pivot_df.fillna('N/A')
    return pivot_df

def establish_dataset_baseline():
    results_manager = ResultsManager.get_manager()
    per_class_samples_for_metric_calc = 10

    experiment_metrics = {}
    dataset_reports = {dataset_name:DatasetReportElements() for dataset_name in list(DATASET_NAMES_AND_SETTINGS.keys())}
    for augmentation in AUGMENTATIONS_OF_INTEREST:
        experiment_metrics[augmentation] = {}
        for dataset_name, db_settings in DATASET_NAMES_AND_SETTINGS.items():
            
            iteration_dict = experiment_metrics[augmentation].setdefault(dataset_name, {})

            db_settings["augmentation_settings"] = AugmentationSettings.create_settings_with_name(augmentation)
            dataset_class = DATA_SET_MODULES.get(dataset_name)
            dataset = dataset_class(**db_settings)
            ds_rprt = dataset_reports.get(dataset_name,DatasetReportElements())
            ds_rprt.add_variant(dataset)

            euclidean_baseline_metric_desc = MetricsDescriptor(metric_calculator=METRIC, dataset=dataset, distance_function=EuclideanDistance(), per_class_samples=per_class_samples_for_metric_calc)
            euclidean_baseline_metrics = euclidean_baseline_metric_desc.calculate_metric()
            iteration_dict["euclidean_distance"] = euclidean_baseline_metrics

            L1_baseline_metric_desc = MetricsDescriptor(metric_calculator=METRIC, dataset=dataset, distance_function=L1Distance(), per_class_samples=per_class_samples_for_metric_calc)
            L1_baseline_metrics = L1_baseline_metric_desc.calculate_metric()
            iteration_dict["L1_distance"] = L1_baseline_metrics

            cubical_complex_baseline_db_settings = copy.deepcopy(db_settings)
            cubical_complex_baseline_db_settings["flatten"] = False
            if dataset_name == "SCEMILA/image_data" or dataset_name == "Acevedo":
                cubical_complex_baseline_db_settings["resize"] = True    
            dataset_cub = dataset_class(**cubical_complex_baseline_db_settings)
            cub_complex_metrics_desc = MetricsDescriptor(metric_calculator=METRIC, dataset=dataset_cub, distance_function=CubicalComplexImageDistanceFunction(), per_class_samples=int(per_class_samples_for_metric_calc / 2))
            cub_complex_metrics = cub_complex_metrics_desc.calculate_metric()
            iteration_dict["cubical_complex_distance"] = cub_complex_metrics

            for transform_name, transform in DEFAULT_TRANSFROM_DICT.items():
                trans_func, trans_settings = transform
                descriptor = EmbeddingDescriptor(f"{dataset_name[0]}_{transform_name}_8", dataset, transform_name, trans_func, trans_settings)
                embeddings, labels, stats_dic = descriptor.generate_embedding_from_descriptor()
                embd_ds = EmbeddingBaseDataset(embedding_id=results_manager.calculate_descriptor_id(descriptor))
                metric_desc = MetricsDescriptor(metric_calculator=METRIC, dataset=embd_ds, distance_function=EuclideanDistance(), per_class_samples=per_class_samples_for_metric_calc)
                iteration_dict[transform_name] = metric_desc.calculate_metric()

            print(cub_complex_metrics_desc.to_dict())
    return experiment_metrics,dataset_reports


def create_pdf_report(dataset_details, metrics_table, hist_file, sample_images_file, conf_matrix_file, dataset_reports,pdf_file_name='experiment_report.pdf'):
    doc = SimpleDocTemplate(pdf_file_name, pagesize=landscape(letter))
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    normal_style = styles['BodyText']

    # Title
    elements.append(Paragraph("Experiment Report", title_style))
    elements.append(PageBreak())

    # Dataset Details
    elements.append(Paragraph("Dataset Details", heading_style))
    for key, value in dataset_details.items():
        elements.append(Paragraph(f"{key}: {value}", normal_style))
    #elements.append(PageBreak())

    # Histogram
    elements.append(Paragraph("Histogram of Dataset", heading_style))
    elements.append(Image(hist_file, width=6 * inch, height=4 * inch))
    #elements.append(PageBreak())

    # Sample Images
    elements.append(Paragraph("Sample Images from Dataset", heading_style))
    elements.append(Image(sample_images_file, width=6 * inch, height=4 * inch))
    #elements.append(PageBreak())

    # Metrics Table
    elements.append(Paragraph("Metrics Table", heading_style))
    table_data = [metrics_table.columns.to_list()] + metrics_table.values.tolist()
    table_data = [[str(item) for item in row] for row in table_data]  # Convert all items to strings
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ])
    table = Table(table_data)
    table.setStyle(table_style)
    elements.append(table)
    #elements.append(PageBreak())

    # Confusion Matrix
    elements.append(Paragraph("Confusion Matrix", heading_style))
    elements.append(Image(conf_matrix_file, width=6 * inch, height=4 * inch))
    #elements.append(PageBreak())
    for dataset_name,dataset_reporter in dataset_reports.items():
        elements.extend(dataset_reporter.get_elements())
    doc.build(elements)

# Example usage:
dataset_details = {
    'Name': 'MNIST',
    'Size': '60,000 images',
    'Classes': '10',
    'Image Size': '28x28 pixels'
}

# Generate dummy metrics for demonstration
experiment_metrics,dataset_reports = establish_dataset_baseline()
metrics_table = create_complete_metrics_table(experiment_metrics)


im_path ="/home/milad/Pictures/find-and-replace-in-postman-2.png"
# Create PDF report
create_pdf_report(dataset_details, metrics_table, im_path, im_path,im_path,dataset_reports)
