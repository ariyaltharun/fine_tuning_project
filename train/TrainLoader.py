from transformers import Trainer,TrainingArguments
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
        args = TrainingArguments(
             f"{args["model"]}-finetuned-lora-food101",
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args["learning_rate"],
            per_device_train_batch_size=args["batch_size"],
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=args["batch_size"],
            fp16=True,
            num_train_epochs=args["num_train_epochs"],
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            label_names=["labels"],
        )
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
