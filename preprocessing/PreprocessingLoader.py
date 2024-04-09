from datasets import DatasetDict
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    DataCollatorWithPadding
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

class ImagePreprocessor:
    def __init__(self, model_checkpoint):
        self.image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
        self.normalize = Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std)
        self.train_transforms = Compose([
            RandomResizedCrop(self.image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            self.normalize,
        ])
        self.val_transforms = Compose([
            Resize(self.image_processor.size["height"]),
            CenterCrop(self.image_processor.size["height"]),
            ToTensor(),
            self.normalize,
        ])

    def preprocess_train(self, example_batch):
        example_batch["pixel_values"] = [self.train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(self, example_batch):
        example_batch["pixel_values"] = [self.val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch


class TextPreprocessing:
    def __init__(self, model_checkpoint) -> None:
        self.preprocessor = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            add_prefix_space=True
        )

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_data_collator(self) -> DataCollatorWithPadding:
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )
        return data_collator
    
    def get_tokenized_dataset(self, dataset: DatasetDict) -> DatasetDict:
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # model.resize_token_embeddings(len(tokenizer))

        def tokenize_function(examples):
            tweets = examples["tweet"]
            preprocess_t= [tweet if isinstance(tweet , str) else str(tweet) for tweet in tweets]
            return self.tokenizer(preprocess_t, truncation= True, padding= "max_length")

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
