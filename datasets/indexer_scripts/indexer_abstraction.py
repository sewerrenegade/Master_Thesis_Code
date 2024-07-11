from abc import ABC, abstractmethod
class Indexer(ABC):
    @abstractmethod
    def get_indexer():
        pass
    @abstractmethod
    def get_per_class_count():
        pass
    @abstractmethod
    def get_random_samples_of_class():
        pass
    @abstractmethod
    def get_instance_level_indicies():
        pass
    @abstractmethod
    def get_bag_level_indicies():
        pass