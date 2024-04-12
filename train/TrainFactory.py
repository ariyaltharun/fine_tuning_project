from .TrainLoader import TrainLoader, TrainLoderbert

class TrainFactory:
    @staticmethod
    def get_trainer(model, args, train_dataset, eval_dataset, tokenizer, compute_metrics, data_collator):
        return TrainLoader(model, args, train_dataset, eval_dataset, tokenizer)
    def get_trainer_bert(model, args, train_dataset, eval_dataset, tokenizer, compute_metrics, data_collator):
        return TrainLoderbert(model, args, train_dataset, eval_dataset, tokenizer)