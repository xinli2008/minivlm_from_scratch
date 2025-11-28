import os
import warnings
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from typing import List
from .model_llm import LLMConfig

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