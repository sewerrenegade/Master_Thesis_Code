    
from io import BytesIO
import os
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.units import inch    
from reportlab.platypus import  Table, TableStyle, Paragraph, Image, PageBreak,Spacer
from PIL import Image as PILImage
import uuid
from reportlab.pdfgen import canvas
import numpy as np
from configs.global_config import GlobalConfig
from reportlab.lib import utils
from results.report_generators.common_report_data import NORMAL_STYLE
from pdf2image import convert_from_path



def pdf_page_to_image(input_pdf, page_number):
    images = convert_from_path(input_pdf, first_page=page_number + 1, last_page=page_number + 1)
    if not images:
        raise ValueError(f"No page found at number {page_number} in the PDF.")
    
    img_path = os.path.join(GlobalConfig.TEMP_RESULTS_FOLDER, f'{uuid.uuid4()}.png')
    images[0].save(img_path, 'PNG')
    return img_path

# Function to append a specific page from another PDF
def append_pdf_page(input_pdf, page_number, output_elements):
    img_path = pdf_page_to_image(input_pdf, page_number)
    img = Image(img_path, width=5 * inch, height=5 * inch)
    output_elements.append(img)

    
def plot_histograms(dict_list):
    histograms = []
    for d in dict_list:
        fig, ax = plt.subplots(figsize=(8, 8))
        keys = list(d.keys())
        values = list(d.values())
        ax.bar(keys, values)
        ax.set_title('Class Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Sample Count')
        plt.xticks(rotation=15, ha='right', fontsize=10)
        histograms.append(fig)
    return histograms
def sort_dict_list(dict_list):
    """
    Sort a list of dictionaries by their keys, treating numeric keys as numbers and non-numeric keys as strings.

    Args:
        dict_list (list): A list of dictionaries to be sorted.

    Returns:
        list: A list of dictionaries with sorted keys.
    """
    sorted_dict_list = []
    for d in dict_list:
        # Convert all keys to strings and sort them
        sorted_dict = {k: d[k] for k in sorted(d.keys(), key=lambda x: int(str(x)) if str(x).isdigit() else str(x))}
        sorted_dict_list.append(sorted_dict)
    
    return sorted_dict_list
def dict_list_to_dataframe(dict_list):
    df = pd.DataFrame(dict_list)
    df = df.fillna(0)  # Replace NaN with 0 for plotting
    return df

def get_histogram_elements(histograms, n_per_row=2):
    elements = []
    # Convert histograms to images and add to table
    img_paths = []
    for i, fig in enumerate(histograms):
        img_path = f'{GlobalConfig.TEMP_RESULTS_FOLDER}{uuid.uuid4()}.png'
        fig.savefig(img_path)
        img_paths.append(img_path)
        plt.close(fig)  # Close the figure to free memory

    # Arrange images in a table
    img_table_data = []
    for i in range(0, len(img_paths), n_per_row):
        img_row = [Image(img_path, width=3 * inch, height=2 * inch) for img_path in img_paths[i:i+n_per_row]]
        if len(img_row) < n_per_row:  # If the last row has fewer images, add spacers
            img_row += [Spacer(3 * inch, 2 * inch)] * (n_per_row - len(img_row))
        img_table_data.append(img_row)

    img_table = Table(img_table_data, hAlign='LEFT')
    elements.append(img_table)
    return elements

def save_tensor_as_png(tensor):
    """
    Save a PyTorch tensor as a PNG image.

    Args:
        tensor (torch.Tensor): Input tensor of shape [1, n, m], [n, m], or [3, n, m].
        output_folder (str): Path where the image will be saved.

    Returns:
        str: Path to the saved image.
    """
    # Ensure the tensor is in the expected shapes
    if tensor.ndim == 3 and tensor.size(0) == 3:
        # RGB image: [3, n, m]
        np_array = tensor.permute(1, 2, 0).numpy()
    elif tensor.ndim == 3 and tensor.size(0) == 1:
        # Single-channel image: [1, n, m]
        np_array = tensor.squeeze(0).numpy()
    elif tensor.ndim == 2:
        # Single-channel image without channel dimension: [n, m]
        np_array = tensor.numpy()
    else:
        raise ValueError("Tensor must have shape [3, n, m], [1, n, m], or [n, m]")

    # If it's a single-channel image, ensure it's properly scaled
    if np_array.ndim == 2:
        np_array = np.expand_dims(np_array, axis=-1)  # Add a dummy channel dimension for compatibility
        np_array = np.repeat(np_array, 3, axis=-1)  # Convert to 3-channel format (greyscale to RGB)

    # Convert to uint8 and save
    np_array = (np_array * 255).astype(np.uint8)  # Scale to [0, 255]
    image = PILImage.fromarray(np_array)

    # Create a unique image path
    img_path = os.path.join(GlobalConfig.TEMP_RESULTS_FOLDER, f'{uuid.uuid4()}.png')
    
    # Save the image
    image.save(img_path)
    
    return img_path


def get_sample_image_elements(image_dict, images_per_row=5):
    elements = []
    data = []
    row = []
    
    for idx, (label, image_path) in enumerate(image_dict.items()):
        if idx > 0 and idx % images_per_row == 0:
            data.append(row)
            row = []
        
        img = utils.ImageReader(image_path)
        aspect = img.getSize()[1] / float(img.getSize()[0])
        img_width = 2 * inch
        img_height = img_width * aspect
        image = Image(image_path, width=img_width, height=img_height)
        caption = Paragraph(str(label), NORMAL_STYLE)
        
        row.append([image, caption])
    
    if row:
        data.append(row)
    
    # Create a table with images and captions
    table_data = []
    for row in data:
        image_row = []
        caption_row = []
        for cell in row:
            image_row.append(cell[0])
            caption_row.append(cell[1])
        table_data.append(image_row)
        table_data.append(caption_row)

    table = Table(table_data)
    table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    
    elements.append(table)
    
    return elements