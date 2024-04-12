from transformers import Trainer
import torch
import evaluate
import numpy as np

metric = evaluate.load("accuracy")

class TrainLoader:
    @staticmethod
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return {"accuracy": (predictions == eval_pred.label_ids).mean()}

    @staticmethod
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        self.trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=self.collate_fn,
        )

    def train(self):
        return self.trainer.train()

    def evaluate(self, val_ds):
        return self.trainer.evaluate(val_ds)

    def get_trainer(self):
        return self.trainer
    
class TrainLoderbert:
    @staticmethod
    def compute_metrics(eval_pred):
        accuracy = evaluate.load("accuracy")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

 

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer,data_collator):
        self.trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
        )

    def train(self):
        return self.trainer.train()

    def evaluate(self, val_ds):
        return self.trainer.evaluate(val_ds)

    def get_trainer(self):
        return self.trainer

