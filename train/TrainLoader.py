from transformers import Trainer,TrainingArguments
import torch
import evaluate
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import  DataLoader, RandomSampler, SequentialSampler
import datetime
import time
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
             f"{args['model']}-finetuned-lora-food101",
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
    
class TrainLoaderbert:
    @staticmethod
    def compute_metrics(eval_pred):
        accuracy = evaluate.load("accuracy")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer,data_collator):
        # args = TrainingArguments(
        #      f"{args['model']}-finetuned-lora",
        #     evaluation_strategy="epoch",
        #     save_strategy="epoch",
        #     learning_rate=args["learning_rate"],
        #     per_device_train_batch_size=args["batch_size"],
        #     per_device_eval_batch_size=args["batch_size"],
        #     num_train_epochs=args["num_train_epochs"],
        #     logging_steps=10,
        #     load_best_model_at_end=True,
        #     metric_for_best_model="accuracy",
        #     label_names=["labels"],
        # )
        args = TrainingArguments(
            output_dir= args['model']+"-lora-text-classification",
            learning_rate=args["learning_rate"],
            per_device_train_batch_size=args["batch_size"],
            per_device_eval_batch_size=args["batch_size"],
            num_train_epochs=args["num_train_epochs"],
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
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

class TrainLoaderCustomLora:
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
    def train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        batch_size = self.args['batch_size']
        train_dataloader = DataLoader(
                    self.train_dataset,  
                    sampler = RandomSampler(self.train_dataset), 
                    batch_size = batch_size 
                )
        validation_dataloader = DataLoader(
                    self.eval_dataset, 
                    sampler = SequentialSampler(self.eval_dataset), 
                    batch_size = batch_size
                )
        optimizer = AdamW(self.model.parameters(),
                  lr = self.args['learning_rate'], 
                  eps = 1e-8
                )
        epochs = self.args['num_train_epochs']
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0,
                                                    num_training_steps = total_steps)
        def flat_accuracy(preds, labels):
            pred_flat = np.argmax(preds, axis=1).flatten()
            labels_flat = labels.flatten()
            return np.sum(pred_flat == labels_flat) / len(labels_flat)
        def format_time(elapsed):
            elapsed_rounded = int(round((elapsed)))
            return str(datetime.timedelta(seconds=elapsed_rounded))
        
        seed_val = 42
        np.random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        training_stats = []
        total_t0 = time.time()
        for epoch_i in range(0, epochs): 
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
            t0 = time.time()
            total_train_loss = 0
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device).squeeze(1) # [batch_size, input_id_size]
                b_input_mask = batch[1].to(device) # [batch_size, input_mask_size]
                b_labels = batch[2].to(device) # [batch_size]
                optimizer.zero_grad()
                output = self.model(b_input_ids, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)        
                loss = output.loss
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)            
            training_time = format_time(time.time() - t0)
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
            # ========================================
            #               Validation
            # ========================================
            print("")
            print("Running Validation...")
            t0 = time.time()
            self.model.eval()
            total_eval_accuracy = 0
            best_eval_accuracy = 0
            total_eval_loss = 0
            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device).squeeze(1) # [batch_size, input_id_size]
                b_input_mask = batch[1].to(device) # [batch_size, input_mask_size]
                b_labels = batch[2].to(device) # [batch_size]
                with torch.no_grad():        
                    output = self.model(b_input_ids, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                loss = output.loss
                total_eval_loss += loss.item()
                logits = output.logits
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            validation_time = format_time(time.time() - t0)
            if avg_val_accuracy > best_eval_accuracy:
                torch.save(self.model, 'bert_model')
                best_eval_accuracy = avg_val_accuracy
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
            return training_stats
