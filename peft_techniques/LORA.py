from abc import (
    ABC,
    abstractmethod
)
from typing import List

from peft import (
    LoraConfig, 
    LoraModel
)
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)


# Abstract Design
class PEFT(ABC):
    """Abstract Class for PEFT Methods
    Attr:
    ------
    + method_name: string

    Methods:
    --------
    + loadPeftConfig() 
    + loadPeftModel(model, config)  # Where will you define peft method name
    """

    # IDK how shud i create constructor or not in abstract class
    # And idk how to create abstract attributes in abstract class
    
    @property
    @abstractmethod
    def peft_method() -> str:
        pass

    @staticmethod
    @abstractmethod
    def loadPeftConfig(config):
        pass

    @staticmethod
    @abstractmethod
    def loadPeftModel(model, config):
        pass



# Adapter Design
class LoRA(PEFT):
    peft_method = "LoRA"

    @staticmethod
    def loadConfig(
            task_type: str,
            rank: int,
            target_modules: List[str],
            lora_alpha: int=8,
            lora_dropout: float=0.0,
        ) -> LoraConfig:
        return LoraConfig(
            task_type=task_type,
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
    
    @staticmethod
    def loadModel(
            model: torch.nn.Module,
            config: LoraConfig
        ) -> LoraModel:
        return LoraModel(
            model,
            config,
            adapter_name="default"
        )  # What is adpdater_name param here???



if __name__ == "__main__":
    model: torch.nn.Module
    lora_config = LoRA.loadConfig(
        task_type="SEQ_CLS",
        rank=4,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=['q_lin','k_lin']
    )
    lora_model = LoRA.loadModel(
        model, 
        lora_config
    )
