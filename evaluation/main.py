import os
import torch
import numpy as np
from typing import Optional, Dict, Any
from dataset import load_dataset
from tqdm import tqdm
from ..hyper_parameters import get_eval_args
from model import load_model_and_tokenizer

class SentimentEvaluator:
    def __init__(self, model: Any, tokenizer_name: Any) -> None:
        self.tokenizer = tokenizer_name
        self.model = model.to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    
    @torch.no_grad()
    def batch_inference(self, texts: [str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return predictions.detach().cpu().numpy()

    def evaluate(self, dataset_split: str = "test") -> None:
        dataset = load_dataset("imdb", split=dataset_split)
        correct = 0
        total = 0
        for i in tqdm(range(0, len(dataset), 32), desc="Evaluating"):
            batch = dataset[i:i+32]
            texts = batch["text"]
            labels = batch["label"]
            predictions = self.batch_inference(texts)
            predicted_labels = np.argmax(predictions, axis=1)
            correct += (predicted_labels == labels).sum()
            total += len(labels)
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__": 
    model_args,finetuning_args =  get_eval_args()
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
    
    evaluator = SentimentEvaluator(model, tokenizer)
    evaluator.evaluate()
