import os
import warnings
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from typing import List
from .model_llm import LLMConfig, LLMModel

warnings.filterwarnings("ignore", category=UserWarning)

class VLMConfig(LLMConfig):
    model_type = "vlm_model"

    def __init__(
            self,
            image_special_token: str = '@'*196,
            image_ids: List = [34] * 196,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_special_token = image_special_token
        self.image_ids = image_ids

class VisionProj(nn.Module):
    def __init__(self, vision_hidden_size: int = 768, llm_hidden_size: int = 512):
        super(VisionProj, self).__init__()
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.vision_proj = nn.Linear(vision_hidden_size, llm_hidden_size)
    
    def forward(self, x):
        return self.vision_proj(x)

class VLMModel(LLMModel):
    config_class = VLMConfig

    def __init__(self, config: VLMConfig = None, vision_model_path: str = "openai/clip-vit-base-patch32", **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.vision_encoder, self.processor = self.get_vision_model(vision_model_path)
        self.vision_proj = VisionProj(llm_hidden_size=self.config.hidden_size)

