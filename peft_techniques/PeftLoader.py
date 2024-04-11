from abc import ABC, abstractmethod
from typing import List, Dict
from peft import LoraConfig, get_peft_model
import torch.nn as nn

# Abstract Base Class for PEFT
class PEFT(ABC):
    @property
    @abstractmethod
    def peft_method() -> str:
        pass

    @staticmethod
    @abstractmethod
    def loadPeftConfig(config: Dict):
        pass

    @staticmethod
    @abstractmethod
    def loadPeftModel(model: nn.Module, config):
        pass

# LoRA Adapter Class
class LoRA(PEFT):
    peft_method = "LoRA"

    @staticmethod
    def loadPeftConfig(config: Dict) -> LoraConfig:
        lora_config = LoraConfig()
        for key, value in config.items():
            if hasattr(lora_config, key):
                setattr(lora_config, key, value)
        return lora_config

    @staticmethod
    def loadPeftModel(model: nn.Module, config: LoraConfig) -> nn.Module:
        lora_model = get_peft_model(model, config)
        return lora_model
