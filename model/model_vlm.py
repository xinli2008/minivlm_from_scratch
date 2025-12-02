import os
import warnings
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from typing import List
from .model_llm import LLMConfig, LLMForCausalLM, MOEFeedForward, CausalLMOutputWithPast
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

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

class VLMModel(LLMForCausalLM):
    config_class = VLMConfig

    def __init__(
            self, 
            config: VLMConfig = None, 
            vision_model_path: str = "openai/clip-vit-base-patch32", 
            **kwargs
            ):
        super().__init__(config, **kwargs)
        self.config = config
        self.vision_encoder, self.processor = self.get_vision_model(vision_model_path)
        self.vision_proj = VisionProj(llm_hidden_size=self.config.hidden_size)

    def get_vision_model(self, vision_model_path: str):
        if not os.path.exists(vision_model_path):
            assert False, f"Vision model path {vision_model_path} does not exist."
        
        model = CLIPModel.from_pretrained(vision_model_path)
        processor = CLIPProcessor.from_pretrained(vision_model_path)

        # NOTE: Freeze vision encoder parameters
        # NOTE: clip模型本身在大规模的图文数据集上训练过, 输出的图片特征已经和文本特征对齐, 因此这里不需要再微调视觉编码器
        for param in model.parameters():
            param.requires_grad = False

        return model.eval(), processor

    def image2tensor(self, image, processor):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")["pixel_values"]
        return inputs

    def get_image_embeddings(self, image_tensor: torch.Tensor, vision_model):
        with torch.no_grad():
            # NOTE：在clip的模型, 第0个token是CLS, 后面的是patch embeddings, 所以取[:, 1:, :]
            image_outputs = vision_model.vision_model(pixel_values=image_tensor).last_hidden_state[:, 1:, :].squeeze()
        return image_outputs
    
    def merge_image_text_embeddings(self, tokens, h, vision_tensors=None):
        """
            将视觉特征插入到文本特征中对应的位置, 替换掉图像占位符的token特征.
            Args:
                tokens: torch.Tensor, [B, seq_len]
                h: torch.Tensor, [B, seq_len, hidden_size]
                vision_tensors: torch.Tensor, [B, num_images, num_patches, vision_hidden_size]
            Returns:
                h: torch.Tensor, [B, seq_len, hidden_size]
            NOTE: 在找mask的时候, 需要找到的是连续的196个图像占位符, 标记为Ture。如果有单独的位置也是图像占位符, 但是不连续, 则不进行替换.
        """
        if vision_tensors is None:
            return h
            
        image_ids = torch.tensor(self.config.image_ids, device=tokens.device)
        image_unique_id = image_ids[0]
        num_image_tokens = len(self.config.image_ids)  # 196
        
        # 找到连续的图像占位符位置
        mask = tokens == image_unique_id  # [B, seq_len]
        
        if not mask.any():
            return h
        
        # 处理每个batch
        batch_size = tokens.shape[0]
        vision_proj = self.vision_proj(vision_tensors)
        vision_proj = vision_proj.squeeze(1) if vision_proj.dim() > 3 else vision_proj
        vision_proj = vision_proj.type_as(h)
        
        for b in range(batch_size):
            batch_mask = mask[b]  # [seq_len]
            if not batch_mask.any():
                continue
                
            # 找到连续的196个True的起始位置
            true_indices = torch.where(batch_mask)[0]
            if len(true_indices) < num_image_tokens:
                continue
                
            # 寻找连续的num_image_tokens个位置
            for i in range(len(true_indices) - num_image_tokens + 1):
                start_idx = true_indices[i]
                end_idx = true_indices[i + num_image_tokens - 1]
                
                # 检查是否连续
                if end_idx - start_idx == num_image_tokens - 1:
                    # 找到连续的196个位置，替换这些位置的特征
                    if b < vision_proj.shape[0]:
                        h[b, start_idx:start_idx + num_image_tokens, :] = vision_proj[b]
                    break
        
        return h

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
            use_cache: bool = False,
            logits_to_keep: int = 0,
            pixel_values: Optional[torch.Tensor] = None,
            **kwargs,
            ):
        _, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"): past_key_values = None

        # NOTE: 通过设置past_key_values=[None]*num_layers来初始化缓存, 然后推理的时候迭代layer的时候传入。
        past_key_values = past_key_values or [None] * self.config.num_hidden_layers
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            if len(pixel_values.shape) == 6:
               pixel_values = pixel_values.squeeze(2)
            bsz, num, _, _, _ = pixel_values.shape
            stack_dim = 1 if bsz > 1 else 0
            vision_tensors = torch.stack([
                self.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                for i in range(num)
            ], dim=stack_dim)
            hidden_states = self.merge_image_text_embeddings(input_ids, hidden_states, vision_tensors=vision_tensors)

        position_embeddings = (
        self.model.freqs_cos[start_pos:start_pos + seq_length],
        self.model.freqs_sin[start_pos:start_pos + seq_length])

        presents = []
        for _, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        output = CausalLMOutputWithPast(logits=logits, past_key_values=presents, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output