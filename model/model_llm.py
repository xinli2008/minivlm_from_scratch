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

