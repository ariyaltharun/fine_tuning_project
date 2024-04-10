from datasets import DatasetDict
from .PreprocessingLoader import (
    ImagePreprocessor,
    TextPreprocessing
)

class PreprocessingFactory:
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint

    def get_preprocessor(self):
        return ImagePreprocessor(self.model_checkpoint)

    def get_text_preprocessor(self, dataset: DatasetDict):
        preprocessing = TextPreprocessing(self.model_checkpoint)
        tokenizer = preprocessing.get_tokenizer()
        datacollator = preprocessing.get_data_collator()
        tokenized_data = preprocessing.get_tokenized_dataset(dataset)
        return (tokenizer, datacollator, tokenized_data)