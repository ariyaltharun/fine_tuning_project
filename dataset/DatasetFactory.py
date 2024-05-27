from .DatasetLoader import (
    ImageDatasetLoader,
    TextDatasetLoader,
    TweetDatasetLoader
)


class DatasetFactory:
    @staticmethod
    def get_dataset(model_type: str):
        if model_type == "vision_transformer":
            loader = ImageDatasetLoader()
            return loader.get_dataset()
        elif model_type == "bert":
            loader = TextDatasetLoader("sg247/binary-classification")
            dataset_tweet = TweetDatasetLoader("Tweet_of_Disaster")
            return loader.get_dataset(),dataset_tweet.get_dataset()
        
        else:
            raise ValueError("Unknown model type")