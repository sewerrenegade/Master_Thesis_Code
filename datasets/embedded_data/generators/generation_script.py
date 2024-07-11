import sys


sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from datasets.embedded_data.generators.embedding_descriptor import EmbeddingDescriptor,SerializableEmbeddingDescriptor,create_serialializable_descriptor_from_live_descriptor
from datasets.embedded_data.generators.generate_embeddings import generate_embedding_from_descriptor
from configs.global_config import GlobalConfig
from datasets.image_augmentor import AugmentationSettings
import umap
import phate
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap,TSNE
import copy
#from datasets.SCEMILA.base_image_SCEMILA import SCEMILAimage_base,SCEMILA_fnl34_feature_base,SCEMILA_DinoBloom_feature_base
from datasets.SCEMILA import *
from datasets.dataset_factory import BASE_MODULES as DATA_SET_MODULES

DEFAULT_TRANSFROM_DICT = {
    "PHATE": (phate.PHATE,{
    'n_components': 3,
    'knn': 10,
    'decay': 40,
    't': 'auto'}),
    
    "TSNE": (TSNE,{
    'n_components': 3,
    'method':'exact'}),
    
    "Isomap": (Isomap,{
    'n_components': 3}),
    
    "UMAP": (umap.UMAP,{
    'n_components': 3}),
    
    "PCA": (PCA,{
    'n_components': 3})
}

SWEEP_PORJECTION_DIM = GlobalConfig.DOWNPROJECTION_TEST_DIMENSIONS

DATASET_NAMES_AND_SETTINGS = {("SCEMILA/image_data","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("SCEMILA/image_data","dino"):{"training_mode":True,"encode_with_dino_bloom":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("Acevedo","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("Acevedo","dino"):{"training_mode":True,"encode_with_dino_bloom":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("FashionMNIST","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("CIFAR10","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("MNIST","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              }
AUGMENTATIONS_OF_INTEREST = ['rotation_aug','all', 'translation_aug','gaussian_noise_aug']
descriptors = []
#dataset = SCEMILA_DINO_BLOOM_BASE_DS
def generate_embeddings():
    for augmentation in AUGMENTATIONS_OF_INTEREST:
        for dataset_name,db_settings in DATASET_NAMES_AND_SETTINGS.items():
            db_settings["augmentation_settings"] = AugmentationSettings.all_false_except_one(augmentation)
            dataset = DATA_SET_MODULES.get(dataset_name[0])
            assert dataset is not None
            dataset = dataset(**db_settings)
            for dim in SWEEP_PORJECTION_DIM:
                for transform_name in list(DEFAULT_TRANSFROM_DICT.keys()):
                    trans_func,trans_settings = DEFAULT_TRANSFROM_DICT[transform_name]
                    trans_settings = copy.deepcopy(trans_settings)
                    trans_settings["n_components"] = dim
                    descriptor = EmbeddingDescriptor(f"{transform_name}_{dim}",dataset,transform_name,trans_func,trans_settings)
                    print(f"started generating embeddings for {descriptor.name}")
                    print(f"finished generating embeddings for {descriptor.name} saved in path {generate_embedding_from_descriptor(descriptor)}")
                    print(create_serialializable_descriptor_from_live_descriptor(descriptor))



if __name__ == '__main__':
    generate_embeddings()
