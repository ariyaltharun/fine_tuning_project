from datasets import DatasetDict
from .PreprocessingLoader import (
    ImagePreprocessor,
    TextPreprocessing,
    tweet_preprocessor
)

class PreprocessingFactory:
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint

    def get_image_preprocessor(self,dataset):
        preprocessing = ImagePreprocessor(self.model_checkpoint)
        train_ds, val_ds = preprocessing.get_train_test_transform(dataset)
        return (train_ds, val_ds)

    def get_text_preprocessor(self, dataset: DatasetDict):
        preprocessing = TextPreprocessing(self.model_checkpoint)
        tokenizer = preprocessing.get_tokenizer()
        datacollator = preprocessing.get_data_collator()
        tokenized_data = preprocessing.get_tokenized_dataset(dataset)
        return (tokenizer, datacollator, tokenized_data)

    def get_tweet_preprocessor(self, dataset: DatasetDict):
        preprocessing = tweet_preprocessor(self.model_checkpoint)
        train_ds,val_ds = preprocessing.get_tokenized_dataset(dataset)
        return (train_ds,val_ds)