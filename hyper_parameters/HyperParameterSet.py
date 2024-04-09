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
        
class QLoRAHyperParameterSet:
    pass

    