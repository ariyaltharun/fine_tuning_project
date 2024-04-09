from abc import ABC, abstractmethod
from datasets import load_dataset
from DatasetLoader import DatasetLoader


class IDatasetLoader(ABC):
    @abstractmethod
    def get_dataset(self):
        pass


class ImageDatasetLoader(DatasetLoader):
    def __init__(self, dataset_name="food101", split="train[:5000]"):
        """
        Initializes the DatasetLoader with the specified dataset name and split configuration.
        :param dataset_name: The name of the dataset to load.
        :param split: The specific split of the dataset to load.
        """
        self.dataset_name = dataset_name
        self.split = split

    def get_dataset(self):
        """
        Loads and returns the dataset based on the initialized configuration.
        :return: The loaded dataset.
        """
        try:
            dataset = load_dataset(self.dataset_name, split=self.split)
            return dataset
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return None


class TextDatasetLoader(DatasetLoader):
    def __init__(self, dataset_name: str, split: str) -> None:
        """
        Initializes the DatasetLoader with the specified dataset name and split configuration.
        :param dataset_name: The name of the dataset to load.
        :param split: The specific split of the dataset to load.
        """
        self.dataset_name = dataset_name
        self.split = split

    def get_dataset(self):
        """
        Loads and returns the dataset based on the initialized configuration.
        :return: The loaded dataset.
        """
        try:
            dataset = load_dataset(self.dataset_name, split=self.split)
            return dataset
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return None
        