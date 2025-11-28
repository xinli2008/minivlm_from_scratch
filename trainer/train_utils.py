"""
训练工具集合
"""

import os
import torch
import torch.nn as nn
import random
import numpy as np
import torch.distributed as dist

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