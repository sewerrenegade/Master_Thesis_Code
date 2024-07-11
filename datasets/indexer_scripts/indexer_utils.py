
from datasets.MNIST.MNIST_indexer import MNIST_Indexer
from datasets.FashionMNIST.FashionMNIST_indexer import FashionMNIST_Indexer
from datasets.CIFAR10.CIFAR10_indexer import CIFAR10_Indexer
from datasets.SCEMILA.SEMILA_indexer import SCEMILA_Indexer
from datasets.Acevedo.acevedo_indexer import AcevedoIndexer

def process_deserialized_json(data):
    if isinstance(data, dict):
        return {int(key): value for key, value in data.items()}
    elif isinstance(data, list):
        return [{int(key): value for key, value in d.items()} for d in data]
    else:
        raise TypeError("Input must be a dictionary or a list of dictionaries")

def get_dataset_indexer(dataset_name):
    return DATASET_TO_INDEXER_MAPPING[dataset_name].get_indexer()



DATASET_TO_INDEXER_MAPPING ={
   "MNIST": MNIST_Indexer,
   "FashionMNIST": FashionMNIST_Indexer,
   "CIFAR10":CIFAR10_Indexer,
   "SCEMILA/image_data":SCEMILA_Indexer,
   "SCEMILA/fnl34_feature_data":SCEMILA_Indexer,
   "SCEMILA_lighting" :SCEMILA_Indexer,
   "Acevedo": AcevedoIndexer
}