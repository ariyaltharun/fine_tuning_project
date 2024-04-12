# Fine Tune Lora Model
import numpy as np
from transformers import TrainingArguments
import evaluate
from dataset import DatasetFactory
from preprocessing import PreprocessingFactory
from model import ModelFactory
from peft_techniques import LoRA
from train import TrainFactory
import hyper_parameters
import train
import model
import dataset
import sys
sys.path.append('/home/pooja/code/hpe/fine_tuning_project-1')
import peft_techniques

model_checkpoint = 'LiyaT3/sentiment-analysis-imdb-distilbert'

# Load Dataset
dataset = DatasetFactory.get_dataset("bert")

# Pre-Process
preprocessor =  PreprocessingFactory(model_checkpoint)
tokenizer, datacollator, tokenized_dataset = preprocessor.get_text_preprocessor(dataset)


# Load Model
model = ModelFactory.create("bert")

# LoRA model
lora_config = LoRA.loadPeftConfig(
    task_type="SEQ_CLS",
    rank=4,
    lora_alpha=32,
    lora_dropout=0.01,
    target_modules=['q_lin','k_lin']
)
lora_model = LoRA.loadModel(
    model, 
    lora_config
)

## Hyper Param
gen_params = {
            "batch_size": 32,
            "learning_rate": 0.01,
            "num_train_epochs": 5,
        }
general_params = hyper_parameters.HyperParameterFactory.get_general_parameters(gen_params)
print(general_params)
args = hyper_parameters.HyperParameterFactory.get_general_parameters(general_params)

#training
trainer = train.TrainFactory.get_trainer(model,args,tokenized_dataset["train"],tokenized_dataset["train"],tokenizer,datacollator)
train_results = trainer.train()

#evaluation
validation_results = trainer.evaluate(tokenized_dataset["train"])