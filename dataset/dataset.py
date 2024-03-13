import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict


def get_imdb_dataset(test_size: float, shuffle: bool=False) -> DatasetDict:
    dataset = load_dataset("imdb", split="train+test")
    return dataset.train_test_split(test_size=test_size, shuffle=shuffle)


class IMDBDataset(Dataset):
    def __init__(self, dataset: DatasetDict, train=True, transform=None, target_transform=None):
        self.dataset = dataset['train'] if train else dataset['test']
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, label = self.dataset[idx].values()
        if self.transform:
            text = self.transform(text)
        if self.target_transform:
            label = self.target_transform(label)
        return text, label

