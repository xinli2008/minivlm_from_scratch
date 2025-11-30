"""
训练工具集合
"""

import os
import torch
import random
import numpy as np
import torch.distributed as dist
from transformers import AutoTokenizer
from torch.utils.data import Sampler
from model.model_vlm import VLMModel

def init_distributed_mode():
    """ 初始化分布式训练环境, 返回当前进程的local_rank"""
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def setup_seed(seed: int):
    """设置随机种子以确保实验的可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

def init_vlm_model(
        vlm_config, 
        from_weight="pretrain_vlm",
        tokenizer_path="",
        vision_model_path="",
        pretrained_model_folder_path="",
        device="cuda",
        freeze_llm=False,
        ):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = VLMModel(config=vlm_config, vision_model_path=vision_model_path)

    if from_weight!="none":
        moe_suffix = "_moe" if vlm_config.use_moe else ""
        weight_path = f"{pretrained_model_folder_path}/{from_weight}_{vlm_config.hidden_size}{moe_suffix}.pth"
        pretrained_model_weights = torch.load(weight_path, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_model_weights, strict=False)
        Logger(f"=> succeed to load pretrained models with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys.")
    
    if freeze_llm:
        for name, param in model.named_parameters():
            if "vision_proj" not in name:
                param.requires_grad = False
        Logger("=> Freeze LLM model parameters.")
        
    Logger(f'所加载VLM Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model.to(device), tokenizer, model.processor

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)

def vlm_checkpoint(
        vlm_config,
        weight="pretrain_vlm",
        model=None,
        optimizer=None,
        epoch=0,
        step=0,
        wandb=None,
        save_dir="../",
        **kwargs):
    pass