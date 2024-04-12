from .TrainLoader import TrainLoader, TrainLoaderbert

class TrainFactory:
    @staticmethod
    def get_trainer(model, args, train_dataset, eval_dataset, tokenizer, compute_metrics=None, data_collator=None):
        return TrainLoader(model, args, train_dataset, eval_dataset, tokenizer)

    @staticmethod
    def get_trainer_bert(model, args, train_dataset, eval_dataset, tokenizer, data_collator=None, compute_metrics=None):
        return TrainLoaderbert(model, args, train_dataset, eval_dataset, tokenizer, data_collator)
