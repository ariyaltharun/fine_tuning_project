from TrainLoader import TrainLoader

class TrainFactory:
    @staticmethod
    def get_trainer(model, args, train_dataset, eval_dataset, tokenizer, compute_metrics, data_collator):
        return TrainLoader(model, args, train_dataset, eval_dataset, tokenizer, compute_metrics, data_collator)
