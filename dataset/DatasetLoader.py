from abc import ABC, abstractmethod
from datasets import load_dataset
import pandas as pd

class IDatasetLoader(ABC):
    @abstractmethod
    def get_dataset(self):
        pass


class ImageDatasetLoader(IDatasetLoader):
    def __init__(self, dataset_name="food101", split = None):
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


class TextDatasetLoader(IDatasetLoader):
    def __init__(self, dataset_name: str, split: str=None) -> None:
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
class TweetDatasetLoader(IDatasetLoader):
    def __init__(self, dataset_name: str, split: str=None) -> None:
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
            dataset = pd.read_csv("../tests/train.csv")
            return dataset
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return None
        
