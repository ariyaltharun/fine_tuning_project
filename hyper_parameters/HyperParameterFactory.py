from HyperParameterSet import LoRAHyperParameterSet,QLoRAHyperParameterSet

class HyperParameterFactory:
    @staticmethod
    def get_peft_parameters(config_name,params=None):
        if config_name.lower() == "lora":
            return LoRAHyperParameterSet.get_parameters(params)
        else:
            raise ValueError("Unknown Peft Technique name provided.")

    def get_general_parameters(self,params):
        return params