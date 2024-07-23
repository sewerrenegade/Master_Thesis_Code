import sys



sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
    
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
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
    "PHATE": (phate.PHATE,{
    'n_components': 8,
    'knn': 10,
    'decay': 40,
    't': 'auto'}),
    
    "TSNE": (TSNE,{
    'n_components': 8,
    'method':'exact'}),
    
    "Isomap": (Isomap,{
    'n_components': 8}),
    
    "UMAP": (umap.UMAP,{
    'n_components': 8}),
    
    "PCA": (PCA,{
    'n_components': 8})
}
METRIC = DistanceMatrixMetricCalculator
SWEEP_PORJECTION_DIM = GlobalConfig.DOWNPROJECTION_TEST_DIMENSIONS

DATASET_NAMES_AND_SETTINGS = {("SCEMILA/image_data"):{"training_mode":True,"balance_dataset_classes": 10,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("Acevedo"):{"training_mode":True,"balance_dataset_classes": 10,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("FashionMNIST"):{"training_mode":True,"balance_dataset_classes": 10,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("CIFAR10"):{"training_mode":True,"balance_dataset_classes": 10,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("MNIST"):{"training_mode":True,"balance_dataset_classes": 10,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              }
AUGMENTATIONS_OF_INTEREST = ['all','none']
descriptors = []
    


def create_complete_metrics_table(experiment_metrics, key_metric='triplet_loss'):
    # Initialize dictionary to store metrics and associated details
    combined_metrics = []

    # Iterate over augmentations of interest
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

    # Create a DataFrame from the combined metrics
    df = pd.DataFrame(combined_metrics)

    # Pivot the DataFrame to have augmentations as columns
    pivot_df = df.pivot_table(index=['Dataset', 'Calculation'], columns=['Augmentation'], values=['KeyMetricValue']).reset_index()

    # Fill NaN values with a placeholder if necessary
    pivot_df = pivot_df.fillna('N/A')

    return pivot_df


def establish_dataset_baseline():
    results_manager = ResultsManager.get_manager()
    per_class_samples_for_metric_calc = 10
    
    experiment_metrics = {}
    for augmentation in AUGMENTATIONS_OF_INTEREST:
        experiment_metrics[augmentation]= {}
        for dataset_name,db_settings in DATASET_NAMES_AND_SETTINGS.items():
            iteration_dict = experiment_metrics[augmentation].setdefault(dataset_name, {})

            db_settings["augmentation_settings"] = AugmentationSettings.create_settings_with_name(augmentation)
            dataset_class = DATA_SET_MODULES.get(dataset_name)
            dataset = dataset_class(**db_settings)
            
            # Euclidean Baseline metrics
            baseline_metric_desc = MetricsDescriptor(metric_calculator=METRIC,dataset=dataset,distance_function= EuclideanDistance(),per_class_samples=per_class_samples_for_metric_calc)
            baseline_metrics = baseline_metric_desc.calculate_metric()
            iteration_dict["euclidean_distance"] = baseline_metrics

            # L1 baseline distance metrics
            baseline_metric_desc = MetricsDescriptor(metric_calculator=METRIC,dataset=dataset,distance_function= L1Distance(),per_class_samples=per_class_samples_for_metric_calc)
            baseline_metrics = baseline_metric_desc.calculate_metric()
            iteration_dict["L1_distance"] = baseline_metrics

            # Cubical Complex metrics
            db_settings_tmp = copy.deepcopy(db_settings)
            db_settings_tmp["flatten"] = False
            if dataset_name == "SCEMILA/image_data" or dataset_name == "Acevedo":
                db_settings_tmp["resize"] = True    
            dataset_cub = dataset_class(**db_settings_tmp)
            cub_complex_metrics_desc = MetricsDescriptor(metric_calculator=METRIC,dataset=dataset_cub,distance_function= CubicalComplexImageDistanceFunction(),per_class_samples=int(per_class_samples_for_metric_calc/2))
            cub_complex_metrics = cub_complex_metrics_desc.calculate_metric()
            iteration_dict["cubical_complex_distance"] = cub_complex_metrics
                
            # Embedding functions metrics
            for transform_name,transform in DEFAULT_TRANSFROM_DICT.items():
                trans_func,trans_settings = transform
                descriptor = EmbeddingDescriptor(f"{dataset_name[0]}_{transform_name}_8",dataset,transform_name,trans_func,trans_settings)
                embeddings,labels,stats_dic = descriptor.generate_embedding_from_descriptor()
                embd_ds = EmbeddingBaseDataset(embedding_id = results_manager.calculate_descriptor_id(descriptor))
                metric_desc = MetricsDescriptor(metric_calculator=METRIC,dataset=embd_ds,distance_function= EuclideanDistance(),per_class_samples=per_class_samples_for_metric_calc)
                iteration_dict[transform_name]= metric_desc.calculate_metric()
                
            print(baseline_metric_desc.to_dict())
    return experiment_metrics

def df_to_pdf2(df,pdf_file_name = 'metrics_results.pdf'):
    num_rows, num_cols = df.shape
    fig_width = num_cols * 2
    fig_height = num_rows * 0.3

    with PdfPages(pdf_file_name) as pdf:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
def df_to_pdf_with_formatting(df, pdf_file_name='metrics_results.pdf'):
    num_rows, num_cols = df.shape
    doc = SimpleDocTemplate(pdf_file_name, pagesize=landscape(letter))

    elements = []

    # Add Title
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    elements.append(Paragraph("Experiment Metrics Report", title_style))

    # Create table data with formatting
    table_data = [df.columns.to_list()] + df.values.tolist()

    # Create Table and apply styles
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

    # Apply specific cell styles
    for i, row in enumerate(df.values):
        for j, cell in enumerate(row):
            if isinstance(cell, str) and cell == 'N/A':
                table_style.add('TEXTCOLOR', (j, i + 1), (j, i + 1), colors.red)
            elif isinstance(cell, float) and cell > 0.9:  # Example condition
                table_style.add('TEXTCOLOR', (j, i + 1), (j, i + 1), colors.green)
                table_style.add('FONTNAME', (j, i + 1), (j, i + 1), 'Helvetica-Bold')
            elif isinstance(cell, float) and cell < 0.5:  # Example condition
                table_style.add('TEXTCOLOR', (j, i + 1), (j, i + 1), colors.blue)
                table_style.add('FONTNAME', (j, i + 1), (j, i + 1), 'Helvetica-Oblique')

    table.setStyle(table_style)

    elements.append(table)

    doc.build(elements)

                        
if __name__ == '__main__':
    experiment_metrics = establish_dataset_baseline()
    metrics_table = create_complete_metrics_table(experiment_metrics, key_metric='knn_acc')
    print(metrics_table)
    # Convert to PDF
    df_to_pdf2(metrics_table, 'baseline.pdf')