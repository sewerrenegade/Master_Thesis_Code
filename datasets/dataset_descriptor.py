from datasets.image_augmentor import DATASET_AUGMENTABIBILITY

class DatasetDescriptor:
    def __init__(self, number_of_channels, dimensions, name, description, is_image, size, test_size, train_size, indexer):
        self.number_of_channels = number_of_channels
        self.dimensions = dimensions
        self.name = name
        self.description = description
        self.augmentation_scheme = DATASET_AUGMENTABIBILITY[name]
        self.is_image = is_image
        self.size = size
        self.test_size = test_size
        self.train_size = train_size
        self.indexer = indexer

    def print_dataset_info(self):
        print(f"Dataset: {self.name}")
        print(f"Description: {self.description}")
        print(f"Number of Channels: {self.number_of_channels}")
        print(f"Dimensions: {self.dimensions}")
        print(f"Augmentation Scheme: {self.augmentation_scheme}")
        print(f"Is Image: {self.is_image}")
        print(f"Size: {self.size}")
        print(f"Test Size: {self.test_size}")
        print(f"Train Size: {self.train_size}")
        print(f"Indexer Type: {type(self.indexer)}")