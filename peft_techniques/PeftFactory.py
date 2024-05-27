from typing import Dict, Type
from .PeftLoader import PEFT, LoRA, Custom_Lora

class PeftFactory:
    peft_methods: Dict[str, Type[PEFT]] = {
        "LoRA": LoRA
        # "OtherPEFTMethod": OtherPEFTMethodClass
    }

    @staticmethod
    def get_peft_method(method_name: str) -> Type[PEFT]:
        if method_name in PeftFactory.peft_methods:
            return PeftFactory.peft_methods[method_name]
        else:
            raise ValueError(f"PEFT method '{method_name}' not recognized")

    @staticmethod
    def create_peft_method(model, config: Dict) -> PEFT:
        config = LoRA.loadPeftConfig(config)
        return LoRA.loadPeftModel(model,config)
    
    @staticmethod
    def get_custom_lora(model,dict):
        model_loaded_with_custom_lora_config = Custom_Lora.final_model(model,dict)
        return model_loaded_with_custom_lora_config
