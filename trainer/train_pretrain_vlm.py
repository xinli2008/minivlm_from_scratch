import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from dataset.vlm_dataset import VLMDataset
from model.model_vlm import VLMConfig
from trainer.train_utils import init_distributed_mode, setup_seed, is_main_process, Logger, init_vlm_model, SkipBatchSampler, vlm_checkpoint, get_lr

def train_epoch(epoch, dataloader, iters, start_step=0, wandb=None):
    loss_function = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    for step, (X, Y, loss_mask, pixel_values) in enumerate(dataloader, start=start_step+1):
        x, y = X.to(args.device), Y.to(args.device)
        loss_mask, pixel_values = loss_mask.to(args.device), pixel_values.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            result = model(x, pixel_values=pixel_values)
            loss = loss_function(result.logits.view(-1, result.logits.size(-1)), y.view(-1)).view(y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += result.aux_loss
            loss = loss/ args.accumulation_steps
        
        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        if step % args.log_interval == 0 or step == iters -1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if vlm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}_epoch{epoch}_iter{step}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # NOTE: 保存权重的时候去掉Clip视觉编码器部分，节省空间
            clean_state_dict = {key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')}
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
            torch.save(clean_state_dict, ckp)
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict, clean_state_dict

        del X, Y, loss_mask, pixel_values, result, loss
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM-Pretrain")
    parser.add_argument("--save_dir", type=str, default="../output", help="保存模型的目录")
    parser.add_argument("--save_weight", default="pretrain_vlm", type=str, help="保存的权重文件名")
    parser.add_argument("--epochs", type=int, default=10, help="训练的轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="每批次的样本数量")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备，例如cuda:0或cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型，例如bfloat16")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积的步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪的阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印的间隔步数")
    parser.add_argument("--save_interval", type=int, default=3000, help="模型保存的间隔步数")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层的大小")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层的数量")
    parser.add_argument('--max_seq_len', default=640, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE（0表示否，1表示是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_data.jsonl", help="预训练数据的路径")
    parser.add_argument("--images_path", type=str, default="../dataset/pretrain_images", help="预训练图像的路径")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb进行日志记录")
    parser.add_argument("--wandb_project", type=str, default="VLM-Pretrain", help="wandb项目名称")

    # resume training args
    parser.add_argument('--from_weight', default='llm', type=str, help="加载的权重文件")
    parser.add_argument('--pretrained_model_folder_path', type=str, default="../pretrained_model", help="预训练模型权重文件夹路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否从断点恢复（0表示否，1表示是）")
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1], help="是否冻结LLM权重（0表示否，1表示是）")

    # tokenizer and vision model paths
    parser.add_argument('--tokenizer_path', type=str, default="../model", help="分词器预训练模型路径")
    parser.add_argument('--vision_model_path', type=str, default="../model/vision_model/clip-vit-base-patch16", help="视觉模型预训练模型路径")

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
    ckpt_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume==1 else None

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
        wandb_run_name = f"miniVLM-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
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
    train_dataset = VLMDataset(
        json_path=args.data_path,
        images_path=args.images_path,
        tokenizer=tokenizer,
        preprocess=preprocess,
        max_length=args.max_seq_len,
        vlm_model=model
    )
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # NOTE: 6. 从ckpt恢复上次训练
    start_epoch, start_step = 0, 0
    if ckpt_data:
        model.load_state_dict(ckpt_data["model"], strict=False)
        optimizer.load_state_dict(ckpt_data["optimizer"])
        scaler.load_state_dict(ckpt_data["scaler"])
        start_epoch = ckpt_data.get("epoch", 0)
        start_step = ckpt_data.get("step", 0)
        Logger(f"=> Resumed from checkpoint: epoch {start_epoch}, step {start_step}")

    # NOTE: 7. DDP封装模型
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # NOTE: 8. 训练循环
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:
            # NOTE: 跳过已经训练过的steps
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_dataset)), args.batch_size, start_step + 1)
            loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=1, pin_memory=True)
            Logger(f'=> Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else:
            # NOTE: 从头开始训练
            loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    Logger("=> Finish Training.")