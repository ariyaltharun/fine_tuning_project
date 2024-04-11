class LoRAHyperParameterSet:
    @staticmethod
    def get_parameters(params=None):
         
        default_params = {
            "r": 16,
            "lora_alpha": 16,
            "target_modules": ["query", "value"], 
            "lora_dropout": 0.1,
            "bias": "none",
            "modules_to_save": ["classifier"],
        }

        if params is None:
            return default_params
        return {**default_params, **params}
class GenHyperParameterSet:
    @staticmethod
    def get_parameters(params=None):
         
        default_params = {
            "remove_unused_columns": False,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "learning_rate": 5e-3,
            "per_device_train_batch_size": 128,
            "gradient_accumulation_steps": 4,
            "per_device_eval_batch_size": 128,
            "fp16": True,
            "num_train_epochs": 10,
            "logging_steps": 10,
            "load_best_model_at_end": True,
            "metric_for_best_model": "accuracy",
            "label_names": ["labels"],
        }

        if params is None:
            return default_params
        return {**default_params, **params}   
        
class QLoRAHyperParameterSet:
    pass

    