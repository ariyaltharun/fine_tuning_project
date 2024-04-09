from transformers import Trainer
import torch

class TrainLoader:
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, compute_metrics, data_collator):
        self.trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

    def train(self):
        return self.trainer.train()

    def evaluate(self, val_ds):
        return self.trainer.evaluate(val_ds)

    def get_trainer(self):
        return self.trainer

    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
