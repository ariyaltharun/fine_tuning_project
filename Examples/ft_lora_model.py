# Fine Tune Lora Model
from transformers import TrainingArguments
import evaluate
from dataset import DatasetFactory
from preprocessing import PreprocessingFactory
from model import ModelFactory
from peft_techniques import LoRA
from train import TrainFactory

model_checkpoint = 'LiyaT3/sentiment-analysis-imdb-distilbert'

# Load Dataset
dataset = DatasetFactory.get_dataset("bert")

# Pre-Process
preprocessor =  PreprocessingFactory(model_checkpoint)
tokenizer, datacollator, tokenized_dataset = preprocessor.get_text_preprocessor(dataset)


# Load Model
model = ModelFactory.create("bert")

# LoRA model
lora_config = LoRA.loadConfig(
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
LEARNING_RATE = 0.01
BATCH_SIZE = 32
NUM_EPOCHS = 5

training_args = TrainingArguments(
    output_dir=model_checkpoint + "-lora-text-classification",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Train
train = TrainFactory.get_trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=datacollator,
    compute_metrics=compute_metrics
)

train.train()
