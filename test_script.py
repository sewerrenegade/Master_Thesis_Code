# from torchvision.datasets import MNIST

# MNIST("data/", train=True, download=True)
# MNIST("data/", train=False, download=True)


# from  models.topology_models.train_reconstruction_distance import train_or_show_rec_autoencoder
# train_or_show_rec_autoencoder()

from gudhi.tensorflow import LowerStarSimplexTreeLayer, CubicalLayer, RipsLayer 
import gudhi 
#import tensorflow as tf
import numpy as np
import pandas as pd


I = np.array(pd.read_csv('data/test/mnist_test.csv', header=None, sep=','), dtype=np.float32)
idx = np.argwhere(I[:,0] == 8)
image = np.reshape(-I[idx[8],1:], [28,28])
image = (image-image.min())/(image.max()-image.min())
image_clean = np.array(image)

X = np.array(image, dtype=np.float32)
layer = CubicalLayer(homology_dimensions=[0])
dgm = layer.call(X)