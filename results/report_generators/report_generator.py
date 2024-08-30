import sys



sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
import pandas as pd
from reportlab.lib.units import inch
from results.report_generators.common_report_functions import append_pdf_page
from results.report_generators.common_report_data import NORMAL_STYLE,HEADING_STYLES,TABLE_STYLE,TITLE_STYLE
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table,  Paragraph, Image, PageBreak
from datasets.SCEMILA import *


descriptors = []


def produce_pivot_table_from_dict_lists(dict_list,index_list,columns,values):
    df = pd.DataFrame(dict_list)
    pivot_df = df.pivot_table(index=index_list, columns=columns, values=values).reset_index()
    pivot_df = pivot_df.fillna('N/A')
    return pivot_df

def produce_element_from_df(pd_df):
    def format_number(value, precision=3):
        """Format a number to the given precision."""
        if isinstance(value, (int, float)):
            return f"{value:.{precision}f}"
        return str(value)
    table_data = [pd_df.columns.to_list()] + pd_df.values.tolist()
    table_data = [[format_number(item) for item in row] for row in table_data]  # Convert all items to strings
    wrapped_table_data = []
    for row in table_data:
        wrapped_row = [Paragraph(cell, NORMAL_STYLE) for cell in row]
        wrapped_table_data.append(wrapped_row)
    
    num_columns = len(pd_df.columns)
    num_rows = len(wrapped_table_data)
    current_pivot_value = None
    color_toggle = False
    # for row_idx in range(1, num_rows):  # Skip header row
    #     pivot_value = wrapped_table_data[row_idx][0].text  # Assuming the first column contains the pivot values
    #     if pivot_value != current_pivot_value:
    #         current_pivot_value = pivot_value
    #         color_toggle = not color_toggle
        
    #     if color_toggle:
    #         bg_color = colors.lightgrey
    #     else:
    #         bg_color = colors.white

    #     for col_idx in range(num_columns):
    #         TABLE_STYLE.add('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), bg_color)
    
    # Define column widths
    table_element = Table(wrapped_table_data)
    table_element.setStyle(TABLE_STYLE)
    return table_element

def create_pdf_from_dataset_reports(dataset_reports):
    elements=[]
    doc = SimpleDocTemplate('experiment_report.pdf', pagesize=landscape(letter))
    elements.append(Paragraph("Experiment Report", TITLE_STYLE))
    elements.append(PageBreak())
    elements.append(Paragraph("Dataset Details", HEADING_STYLES[2]))
    elements.append(PageBreak())
    for dataset_name,dataset_reporter in dataset_reports.items():
        elements.extend(dataset_reporter.get_elements())
    doc.build(elements)
    
    
def create_pdf_report(metrics_table,dataset_reports,pdf_file_name='experiment_report.pdf'):
    doc = SimpleDocTemplate(pdf_file_name, pagesize=landscape(letter))
    elements = []
    


    # Title
    elements.append(Paragraph("Experiment Report", TITLE_STYLE))
    elements.append(PageBreak())
    append_pdf_page("/home/milad/Desktop/TUM_UNI/ML_in_geo_v2/L1.pdf",1,elements)

    # Dataset Details
    elements.append(Paragraph("Dataset Details", HEADING_STYLES[2]))
    #elements.append(PageBreak())
    for dataset_name,dataset_reporter in dataset_reports.items():
        elements.extend(dataset_reporter.get_elements())

    elements.append(Paragraph("Metrics Table", HEADING_STYLES[2]))
    
    table_data = [metrics_table.columns.to_list()] + metrics_table.values.tolist()
    table_data = [[str(item) for item in row] for row in table_data]  # Convert all items to strings
    table_style = TABLE_STYLE
    table = Table(table_data)
    #elements.append(PageBreak())


    # Metrics Table
    table.setStyle(table_style)
    elements.append(table)
    #elements.append(PageBreak())

    doc.build(elements)

# # Generate dummy metrics for demonstration
# experiment_metrics,dataset_reports = establish_baseline()
# metrics_table = create_complete_metrics_table(experiment_metrics)


# im_path ="/home/milad/Pictures/find-and-replace-in-postman-2.png"
# # Create PDF report
# create_pdf_report( metrics_table, dataset_reports)
