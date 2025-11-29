import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_vlm import VLMConfig, VLMModel
from trainer.train_utils import init_distributed_mode, setup_seed, is_main_process, Logger, init_vlm_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM-Pretrain")
    parser.add_argument("--save_dir", type=str, default="./output", help="保存模型的目录")
    parser.add_argument("--save_weight", default="pretrain_vlm", type=str, help="保存的权重文件名")
    parser.add_argument("--epochs", type=int, default=10, help="训练的轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="每批次的样本数量")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备，例如cuda:0或cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型，例如bfloat16")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积的步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪的阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印的间隔步数")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存的间隔步数")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层的大小")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层的数量")
    parser.add_argument('--max_seq_len', default=640, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE（0表示否，1表示是）")
    parser.add_argument("--data_path", type=str, default="/home/lixin/workspace/personal_learning/minivlm_from_scratch/dataset/pretrain_data.jsonl", help="预训练数据的路径")
    parser.add_argument("--images_path", type=str, default="/home/lixin/workspace/personal_learning/minivlm_from_scratch/dataset/pretrain_images", help="预训练图像的路径")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb进行日志记录")
    parser.add_argument("--wandb_project", type=str, default="VLM-Pretrain", help="wandb项目名称")

    # resume training args
    parser.add_argument('--from_weight', default='llm', type=str, help="加载的权重文件")
    parser.add_argument('--pretrained_model_folder_path', type=str, default="/home/lixin/workspace/personal_learning/minivlm_from_scratch/pretrained_model", help="预训练模型权重文件夹路径")
    parser.add_argument('--from_resume', default=1, type=int, choices=[0, 1], help="是否从断点恢复（0表示否，1表示是）")
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1], help="是否冻结LLM权重（0表示否，1表示是）")

    # tokenizer and vision model paths
    parser.add_argument('--tokenizer_path', type=str, default="/home/lixin/workspace/personal_learning/minivlm_from_scratch/model", help="分词器预训练模型路径")
    parser.add_argument('--vision_model_path', type=str, default="/home/lixin/workspace/personal_learning/minivlm_from_scratch/model/vision_model/clip-vit-base-patch16", help="视觉模型预训练模型路径")

    args = parser.parse_args()

    # NOTE: 1. 初始化环境和随机种子
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    Logger(f"=> Using device: {args.device}")

    # NOTE: 2. 配置目录、模型参数、检查ckpt
    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, max_seq_len=args.max_seq_len, use_moe=bool(args.use_moe))
    # ckpt_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume==1 else None

    # NOTE: 3. 设置混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # NOTE: 4. 配置wandb或者tensorboard
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckpt_data.get('wandb_id') if ckpt_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-V-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # NOTE: 5. 定义模型、数据、优化器
    model, tokenizer, preprocess = init_vlm_model(
        vlm_config=vlm_config,
        from_weight=args.from_weight,
        tokenizer_path=args.tokenizer_path,
        vision_model_path=args.vision_model_path,
        pretrained_model_folder_path=args.pretrained_model_folder_path,
        device=args.device,
        freeze_llm=bool(args.freeze_llm),
    )

    # NOTE: 6. 从ckpt恢复上次训练

    # NOTE: 7. DDP封装模型

    # NOTE: 8. 训练循环