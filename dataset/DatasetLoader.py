from abc import ABC, abstractmethod

class IDatasetLoader(ABC):
    @abstractmethod
    def get_dataset(self):
        pass
