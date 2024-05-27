from datasets import DatasetDict, Value
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    DataCollatorWithPadding
)
import torch
from torch.utils.data import TensorDataset, random_split

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup,DistilBertTokenizer

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

    def get_train_test_transform(self,dataset):
        splits = dataset.train_test_split(test_size=0.1)
        train_ds = splits["train"]
        val_ds = splits["test"]
        train_ds.set_transform(self.preprocess_train)
        val_ds.set_transform(self.preprocess_val)
        return train_ds, val_ds

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
        # The model we use accepts label dtype `int`
        dataset = dataset.cast_column('label', Value(dtype='int32'))

        # Filter NULL/None rows in the dataset
        def remove_null(x):
            if x['tweet'] is not None and x['label'] is not None:
                return x
            return None

        dataset = dataset.filter(remove_null)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # model.resize_token_embeddings(len(tokenizer))

        def tokenize_function(examples):
            tweets = examples["tweet"]
            preprocess_t= [tweet if isinstance(tweet , str) else str(tweet) for tweet in tweets]
            return self.tokenizer(preprocess_t, truncation= True, padding= "max_length")

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset


class tweet_preprocessor:
        def __init__(self, model_checkpoint) -> None:
            self.preprocessor = None
            self.tokenizer =  DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        
        def get_tokenizer(self):
                return self.tokenizer

        def get_tokenized_dataset(self, dataset: DatasetDict) -> DatasetDict:
            tweets = dataset.text.values
            labels = dataset.target.values
            input_ids = []
            attention_masks = []
            max_len = 0
            for sent in tweets:
                input_ids = self.tokenizer.encode(sent, add_special_tokens=True)
                max_len = max(max_len, len(input_ids))
            for tweet in tweets:
                encoded_dict = self.tokenizer.encode_plus(
                                    tweet,                     
                                    add_special_tokens = True, 
                                    max_length = max_len,      
                                    pad_to_max_length = True,
                                    return_attention_mask = True,
                                    return_tensors = 'pt',     
                            )
   
                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
            input_ids = torch.cat(input_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)
            labels = torch.tensor(labels)
            dataset = TensorDataset(input_ids, attention_masks, labels)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset)  - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            return train_dataset, val_dataset

