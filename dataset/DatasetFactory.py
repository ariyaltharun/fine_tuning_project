from fine_tuning_project.dataset.DatasetLoader import ImageDatasetLoader
from DatasetLoader import DatasetLoader
class DatasetFactory:
    @staticmethod
    def get_dataset(model_type: str) -> DatasetLoader:
        if model_type == "vision_transformer":
            loader = ImageDatasetLoader()
            return loader.get_dataset()
        else:
            raise ValueError("Unknown model type")