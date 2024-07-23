from abc import ABC, abstractmethod

class Metric(ABC):
    @abstractmethod
    def get_settings():
        pass
    @abstractmethod
    def calculate_metric():
        pass
    
