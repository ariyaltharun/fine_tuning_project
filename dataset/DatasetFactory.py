from .DatasetLoader import (
    DatasetLoader,
    ImageDatasetLoader,
    TextDatasetLoader
)


class DatasetFactory:
    @staticmethod
    def get_dataset(model_type: str) -> DatasetLoader:
        if model_type == "vision_transformer":
            loader = ImageDatasetLoader()
            return loader.get_dataset()
        elif model_type == "bert":
            loader = TextDatasetLoader("sg247/binary-classification")
            return loader.get_dataset()
        else:
            raise ValueError("Unknown model type")