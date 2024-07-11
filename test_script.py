# from torchvision.datasets import MNIST

# MNIST("data/", train=True, download=True)
# MNIST("data/", train=False, download=True)

# #----------------------------------------------------------------------------------------------
# from  models.topology_models.train_reconstruction_distance import train_or_show_rec_autoencoder
# train_or_show_rec_autoencoder('models/topology_models/reconstruction_distance_parameters/MNIST_Reconstruction_model_32_150.pth')

# from gudhi.tensorflow import LowerStarSimplexTreeLayer, CubicalLayer, RipsLayer 
# import gudhi 
# #import tensorflow as tf
# import numpy as np
# import pandas as pd


# I = np.array(pd.read_csv('data/test/mnist_test.csv', header=None, sep=','), dtype=np.float32)
# idx = np.argwhere(I[:,0] == 8)
# image = np.reshape(-I[idx[8],1:], [28,28])
# image = (image-image.min())/(image.max()-image.min())
# image_clean = np.array(image)

# X = np.array(image, dtype=np.float32)
# layer = CubicalLayer(homology_dimensions=[0])
# dgm = layer.call(X)
# #--------------------------------------------------------------------------------------------
# from  models.topology_models.visualisation.test_embeddings import calculate_class_distance_embedded_MNIST, visualize_array

# mean, var = calculate_class_distance_embedded_MNIST()
# visualize_array(zip(mean,var))
#--------------------------------------------------------------------------------------------
# from  distance_functions.input_distance_function_metrics.test_distances import calculate_class_distance_MNIST, visualize_array

# mean, var = calculate_class_distance_MNIST()
# visualize_array(zip(mean,var))
# #--------------------------------------------------------------------------------------------


# configs = [
#     #["configs/test/","topo_test_reconstruction.yaml"],
    
#     #["configs/test/","test.yaml"],
#     ["configs/test","topo_test_reconstruction.yaml"],
#     ]
# from fine_tune.wandb_sweeper import sweep_parameters
# from train import main,set_config_file_environment_variable,initialize_config_env
# initialize_config_env()
# set_config_file_environment_variable(configs[0][0],configs[0][1])

# main()

# from datasets.embedded_data.dataset.embedding_base import baseDataset

# baseDataset("data/MNIST/embeddings/UMAP/")


# import torchvision.transforms as transforms

# t1 = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
# t2 = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])

# print(t1==t2)
# #---------------------------------------------------------------------------------------------------------------
# import pandas as pd
# import pandas as pd

# # Function to remove leading zeros
# def remove_leading_zeros(filename):
#     # Split the filename and extension
#     name, ext = filename.split('.')
#     # Convert name to an integer to remove leading zeros, then convert back to string
#     name = str(int(name))
#     # Recombine name and extension
#     return f'{name}.{ext}'

# # Load the CSV file into a DataFrame
# df = pd.read_csv('data/SCEMILA/meta_files/image_annotation_master.csv')

# # Apply the function to the DataFrame (assuming the filenames are in a column named 'filenames')
# df['im_tiffname'] = df['im_tiffname'].apply(remove_leading_zeros)

# # Save the modified DataFrame back to a CSV file if needed
# df.to_csv('data/SCEMILA/meta_files/non_zero_image_annotation_master.csv', index=False)

# import re
# # Regular expression to match the pattern
# pattern = re.compile(r"(.*\/)image_(\d+)\.tif$")
# path = "data/SCEMILA/image_data/CBFB_MYH11/AQK/image_021.tif"
# match = pattern.match(path)
# if match:
#     directory_path = match.group(1)
#     integer_value = int(match.group(2))
#     print(directory_path)
#     print(integer_value)
# else:
#     raise ValueError("The provided path does not match the expected pattern.")
# import torchvision
# testset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=False)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=False)

# import torch
# import matplotlib.pyplot as plt
# from torchvision import transforms
# import torchvision
# from PIL import Image
# import numpy as np
# from datasets.image_augmentor import IMAGE_AUGMENTATION_TRANSFORM_LIST

# import tifffile as tiff

# # Function to add Gaussian noise to an image
# def add_gaussian_noise(image, mean=0, std=0.02):
#     noise = torch.randn(image.size()) * std + mean
#     noisy_image = image + noise
#     noisy_image = torch.clamp(noisy_image, 0, 1)  # Clamping to keep pixel values in [0, 1]
#     return noisy_image

# # Define a series of transformations including rotation, shift, and noise addition
# transform = transforms.Compose(IMAGE_AUGMENTATION_TRANSFORM_LIST)
# for i in range(100):
#     # Load an example image
#     image_path = f'data/SCEMILA/image_data/NPM1/ALA/image_{i}.tif'  # Replace with your image path
#     image = tiff.imread(image_path)
#     image = Image.fromarray(image)
#     image = Image.open(image_path).convert('RGB')
#     # Apply the transformation
#     #tensor_image = transforms.ToTensor()(image)
#     augmented_image = transform(image)

#     # Convert back to PIL Image for display
#     #augmented_image_pil = transforms.ToPILImage()(augmented_image)

#     # Display the original and augmented images
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title('Original Image')
#     plt.imshow(image)
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.title('Augmented Image')
#     plt.imshow(augmented_image)
#     plt.axis('off')

#     plt.show()

# from datasets.SCEMILA import *
# x = SCEMILA_Indexer()
# pass

from datetime import datetime

# Get the current date and time
current_time = datetime.now()

# Truncate seconds and microseconds to get up to the minute
current_time_up_to_minute = current_time.replace(microsecond=0)

# Print the current time up to the minute
print(current_time_up_to_minute)