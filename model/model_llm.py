import math
from transformers import PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class LLMConfig(PretrainedConfig):
    model_type = "llm_model"
    def __init__(
            self,
            dropout: float=0.0,
            bos_token_id: int=1,
            eos_token_id: int=2,
            hidden_act: str="silu",
            hidden_size: int=512,
            intermediate_size: int=None,
            max_position_embeddings: int=32768,
            num_attention_heads: int=8,
            num_hidden_layers: int=8,
            num_key_value_heads: int=2,
            vocab_size: int=6400,
            rms_norm_eps: float=1e-6,
            rope_theta: float=10000000.0,
            inference_rope_scaling: bool=False,
            flash_attn: bool = True,
            # Add some specific configurations of MOE
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = "softmax",
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_probs: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling

        # NOTE: 外推长度 = factor * original_max_position_embeddings
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn

        # MOE configurations
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_function = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_probs = norm_topk_probs

class RMSNorm(torch.nn.Module):
    """
        RMSNorm implementation, Root Mean Square Layer Normalization, 均方根归一化.
        传统的LayerNorm是基于均值和标准差进行归一化的,而RMSNorm仅基于均方根进行归一化。
        相对于传统的LayerNorm, RMSNorm减少了均值计算和方差计算的操作。
        公式:  output = weight * (x / sqrt(mean(x^2) + eps))
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def rotate_half(x):
    """
        对张量的最后一个维度进行旋转操作
        此函数实现的是"头尾配对"方式的旋转,将向量的前半部分与后半部分进行交叉旋转。具体来说,对于最后一个维度的元素,前半部分的每个元素与后半部分对应位置的元素组成旋转对,用于后续的复数旋转计算。
        对于输入 [q0, q1, q2, q3, q4, q5]，输出为 [-q3, -q4, -q5, q0, q1, q2]。
    """
    x1 = x[..., : x.shape[-1] // 2] 
    x2 = x[..., x.shape[-1] // 2 :] 
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
        应用旋转位置嵌入到Query和Key上。
        q * cos = [q0*cosθ0, q1*cosθ1, q2*cosθ2, q3*cosθ0, q4*cosθ1, q5*cosθ2]
        rotate_half(q) * sin = [-q3*sinθ0, -q4*sinθ1, -q5*sinθ2, q0*sinθ0, q1*sinθ1, q2*sinθ2]
    """
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复Key或Value张量以匹配Query的头数"""
    batch_size, n_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x.unsqueeze(2).expand(batch_size, n_heads, n_rep, seq_len, head_dim).contiguous().view(batch_size, n_heads * n_rep, seq_len, head_dim)
    return x

def precompute_freqs_cos_sin(
        dim: int,
        max_position_embeddings: int,
        rope_base: float = 10000000.0,
        rope_scaling: Optional[dict] = None,
):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    if rope_scaling is not None:
        # We implement YARN(You Ain't RoPE Yet) scaling method
        pass

    t = torch.arange(max_position_embeddings, dtype=freqs.dtype)
    # NOTE: outter?
    freqs = torch.outer(t, freqs).float()  # [max_position_embeddings, dim/2]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # [max_position_embeddings, dim]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # [max_position_embeddings, dim]
    return freqs_cos, freqs_sin

class Attention(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        assert config.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = config.num_attention_heads
        # NOTE: GQA, Grouped Query Attention, 每组Query对应一组KV, 减少计算和内存开销。
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, self.n_local_heads * self.head_dim, bias = False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias = False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias = False)
        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, config.hidden_size, bias = False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attn
        print("=> Flash Attention Enabled" if self.flash else "Flash Attention Disabled")

    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, embed_dim = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim).transpose(1, 2)     # [batch_size, n_local_heads, seq_len, head_dim]
        xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_local_kv_heads, seq_len, head_dim]
        xv = xv.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_local_kv_heads, seq_len, head_dim]

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # NOTE: kv_cache implementation
        if past_key_value is not None:
            x_k = torch.cat([past_key_value[0], xk], dim=2)
            x_v = torch.cat([past_key_value[1], xv], dim=2)
        past_kv = (x_k, x_v) if use_cache else None

        # NOTE: GQA implementation
        xq, xk, xv = (xq, repeat_kv(xk, self.n_rep), repeat_kv(xv, self.n_rep))

        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attention_mask = (
                None
                if attention_mask is None
                else attention_mask.view(batch_size, 1, 1, -1).expand(batch_size, self.n_local_heads, seq_len, -1).bool()
            )
            # NOTE: is_casual: 表示这是一个因果(casual)或自回归(auto-regressive)的注意力机制,即每个位置只能关注当前位置及之前的位置,不能关注未来的位置。
            # NOTE: 在代码的具体实现中, 主要是先创建一个全1的attention_mask, 然后通过torch.tril函数生成一个下三角矩阵, 最后将其转换为布尔类型。
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # NOTE: 注意力机制的普通实现
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.tril(torch.full((seq_len, seq_len), float('-inf'), device=scores.device), diagonal=1).unsqueeze(0).unsqueeze(0)
        
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
    
class FeedForward(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # NOTE: ACT2FN: 激活函数映射表,使用的时候通过ACT2FN[config.hidden_act]来获取具体的激活函数
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class LLMBlock(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
    
class LLMModel(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([LLMBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cos_sin(
            dim = config.hidden_size // config.num_attention_heads,
            max_position_embeddings = config.max_position_embeddings,
            rope_base = config.rope_theta,
            rope_scaling = config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs):
        _, seq_len = input_ids.shape
        if hasattr(past_key_values, "layers"): past_key_values = None

        # NOTE: 通过设置past_key_values=[None]*num_layers来初始化缓存, 然后推理的时候迭代layer的时候传入。
        past_key_values = past_key_values or [None] * self.num_hidden_layers
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # NOTE: input_ids to embeddings
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_len],
            self.freqs_sin[start_pos:start_pos + seq_len]
        )

        presents = []
        for layer_idx, (layer, past_key_values) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                past_key_value=past_key_values,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states, presents
    
class LLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LLMConfig

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model = LLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # NOTE: 在语言模型中, 通常会共享输入嵌入层(embed_tokens)和输出层(lm_head)的权重。
        # 这种做法的主要目的是减少模型参数量, 同时提高模型的泛化能力。共享权重的假设是, 输入嵌入和输出嵌入在语义空间中应该是对称的。
        self.model.embed_tokens.weight = self.lm_head.weight
        self.out = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT