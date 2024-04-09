from HyperParameterSet import LoRAHyperParameterSet

class HyperParameterFactory:
    @staticmethod
    def get_hyper_parameters(config_name):
        if config_name.lower() == "lora":
            return LoRAHyperParameterSet.get_parameters()
        else:
            raise ValueError("Unknown model name provided.")
