# from datasets import load_dataset, DatasetDict, Dataset

# from transformers import (
#     AutoTokenizer,
#     AutoConfig,
#     AutoModelForSequenceClassification,
#     DataCollatorWithPadding,
#     TrainingArguments,
#     Trainer)

# from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
# import evaluate
# import torch
# import numpy as np

# model_checkpoint = 'LiyaT3/sentiment-analysis-imdb-distilbert'

# id2label = {0: "Negative", 1: "Positive"}
# label2id = {"Negative":0, "Positive":1}

# model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     model.resize_token_embeddings(len(tokenizer))

# def tokenize_function(examples):
#     text = examples["sentence"]

#     #tokenize and truncate text
#     tokenizer.truncation_side = "left"
#     tokenized_inputs = tokenizer(
#         text,
#         return_tensors="np",
#         truncation=True,
#         max_length=512
#     )

#     return tokenized_inputs

# tokenized_dataset = dataset.map(tokenize_function, batched=True)
# tokenized_dataset

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# # training 


# peft_config = LoraConfig(task_type="SEQ_CLS",
#                         r=4,
#                         lora_alpha=32,
#                         lora_dropout=0.01,
#                         target_modules = ['query'])

# peft_config

# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

# # hyperparameters

# lr = 
# batch_size = 
# num_epochs = 

# training_args = TrainingArguments(
#     output_dir= model_checkpoint + "-lora-text-classification",
#     learning_rate=lr,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=num_epochs,
#     weight_decay=0.01,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["validation"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics, # add evaluation fun
# )

# trainer.train()

# # prediction 


# model.to('cpu')

# print("Trained model predictions:")
# print("--------------------------")
# for text in text_list:
#     inputs = tokenizer.encode(text, return_tensors="pt").to("cpu")

#     logits = model(inputs).logits
#     predictions = torch.max(logits,1).indices

#     print(text + " - " + id2label[predictions.tolist()[0]])
