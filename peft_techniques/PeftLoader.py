from functools import partial
from abc import ABC, abstractmethod
from typing import List, Dict
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch

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

# Custom Lora Layer Implementation
class LoRALayer(torch.nn.Module):
        def __init__(self, in_dim, out_dim, rank, alpha):
            super().__init__()
            std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
            self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
            self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
            self.alpha = alpha

        def forward(self, x):
            x = self.alpha * (x @ self.A @ self.B)
            return x
        
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)



class Custom_Lora(PEFT):
    def final_model(model,dict):
        lora_r = dict['r']
        lora_alpha = dict['lora_alpha']
        lora_dropout = dict['lora_dropout']
        lora_query = True
        lora_key = True
        lora_value = True
        lora_projection = True
        lora_mlp = True
        lora_head = True
        layers = []
        assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
        for layer in model.distilbert.transformer.layer:
            if lora_query:
                layer.attention.q_lin = assign_lora(layer.attention.q_lin)
            if lora_key:
                layer.attention.k_lin = assign_lora(layer.attention.k_lin)
            if lora_value:
                layer.attention.v_lin = assign_lora(layer.attention.v_lin)
            if lora_projection:
                layer.attention.out_lin = assign_lora(layer.attention.out_lin)
            if lora_mlp:
                layer.ffn.lin1 = assign_lora(layer.ffn.lin1)
                layer.ffn.lin2 = assign_lora(layer.ffn.lin2)
            if lora_head:
                model.pre_classifier = assign_lora(model.pre_classifier)
                model.classifier = assign_lora(model.classifier)

        return model
