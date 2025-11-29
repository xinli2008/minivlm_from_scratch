import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from model.model_vlm import VLMModel

class VLMDataset(Dataset):
    def __init__(self, json_path, images_path, tokenizer, preprocess=None, max_length=512, image_special_token="@"*196):
        pass
        

    