from transformers import AutoImageProcessor
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
