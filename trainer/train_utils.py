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
import math

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
        
    Logger(f'=> Trainable parameter in VLM is {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} million')
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

def get_lr(current_step, total_steps, lr):
    """
        学习率调度函数，采用余弦退火策略。
        余弦退火策略主要是通过在训练过程中逐渐降低学习率，从而帮助模型更好地收敛，避免在训练后期出现震荡现象。
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def vlm_checkpoint(vlm_config, weight='pretrain_vlm', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if vlm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}_resume.pth'
    
    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        
        # NOTE: 移除vision_encoder参数, 因为Clip视觉编码器本身就没有训练
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        
        # NOTE: 保存resume文件用于断点续训
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, clean_state_dict, resume_data
        torch.cuda.empty_cache()
    else:  
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None