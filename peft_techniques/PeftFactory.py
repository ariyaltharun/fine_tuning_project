from typing import Dict, Type
from .PeftLoader import PEFT, LoRA

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
