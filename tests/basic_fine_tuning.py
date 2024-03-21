from datasets import load_dataset
import evaluate
import numpy as np
import accelerate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer,AutoTokenizer
from transformers import DataCollatorWithPadding

dataset = load_dataset("sg247/binary-classification")
print(dataset)

train, test = dataset['train'],dataset['test']
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

def preprocess_function(examples):
    tweets = examples["tweet"]
    processed_tweets = [tweet if isinstance(tweet, str) else str(tweet) for tweet in tweets]
    return tokenizer(processed_tweets, truncation=True, padding="max_length")

tokenized_data = dataset.map(preprocess_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="basic_model_fine_tuned",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


text = "Mann that's look awesome"
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="basic_model_fine_tuned")
classifier(text)