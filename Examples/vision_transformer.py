import sys
sys.path.append('S:/SiDhU/Codes/HP/actual_repo/fine_tuning_project/')
import dataset
import model
import hyper_parameters
import train
import preprocessing

model_name = "vision_transformer"
model_checkpoint = "google/vit-base-patch16-224-in21k"
data = dataset.DatasetFactory.get_dataset(model_name)
parameter = {
            "r": 16,
            "lora_alpha": 16,
            "target_modules": ["query", "value"], 
            "lora_dropout": 0.1,
            "bias": "none",
        }
gen_params = {
            "batch_size": 128,
            "learning_rate": 0.01,
            "num_train_epochs": 2,
        }

peft_params = hyper_parameters.HyperParameterFactory.get_peft_parameters("lora",parameter)
general_params = hyper_parameters.HyperParameterFactory.get_general_parameters(gen_params)
print(peft_params)
print(general_params)
vision_model = model.ModelFactory.create(model_name, data)
processor = preprocessing.PreprocessingFactory(model_checkpoint)
train_ds, val_ds = processor.get_image_preprocessor(data)

# --------------------------------------------------------------------------------
# We have done till here 
# --------------------------------------------------------------------------------
# This we have to add to the peft_techniques
# --------------------------------------------------------------------------------

from peft import LoraConfig, get_peft_model

config = LoraConfig()
for key, value in parameter.items():
    if hasattr(config, key): 
        setattr(config, key, value)
lora_model = get_peft_model(model, config)

# ----------------------------------------------------------------------------
# This fucntion I think we can add to the evaluation modules
# ----------------------------------------------------------------------------

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

print_trainable_parameters(lora_model)
# --------------------------------------------------------------------------------
# This I thought we have defined in the train module but in that it is directly pickking up 
# we need to figure out where to add this
# --------------------------------------------------------------------------------

# from this training args I have defined "batch_size": 128 "learning_rate": 0.01, "num_train_epochs": 2, which we will
# get from the user rest other parameter letter we can get from the user

# --------------------------------------------------------------------------------

from transformers import TrainingArguments, Trainer


model_name = model_checkpoint.split("/")[-1]
batch_size = 128

args = TrainingArguments(
    f"{model_name}-finetuned-lora-food101",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=10,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    label_names=["labels"],
)
# ---------------------------------------------------------------------------------------------
# This we have not define anywhere will have to figure out where to add
# ---------------------------------------------------------------------------------------------
import numpy as np
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)



# -------------------------------------------------------------------------------------
# This as much as I remember is defined inside the trian module
# -------------------------------------------------------------------------------------


import torch
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}



trainer = Trainer(
    lora_model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()


validation_results = trainer.evaluate(val_ds)


# That's it peace 🥳🥳