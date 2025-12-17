"""
Time-Wise Depth HelixNet Model Implementation (v4 - "Free Lunch" Edition)

Updates:
1. Sliding Window Attention: Uses Flash Attention's window_size (or a manual band mask)
   to reduce complexity from O(N^2) to O(N).
2. Decoupled Query: Separate projection for Whiteboard Reading (Reasoning) vs. 
   Attention Querying (Copying).
3. Learnable Aggregation: The Global Whiteboard state is now a learnable weighted sum 
   of layer outputs, rather than a simple mean.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.utils import logging
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from configuration_depth_helixnet import HelixNetConfig

# Attempt to import Flash Attention for the Sliding Window optimization
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

logger = logging.get_logger(__name__)


# =============================================================================
# 1. The Global Whiteboard Cache
# =============================================================================

@dataclass
class WhiteboardCache(Cache):
    """
    A unified cache that holds:
    1. Standard Attention KV Cache
    2. The Global Whiteboard State Matrix
    """
    key_cache: List[torch.Tensor] = field(default_factory=list)
    value_cache: List[torch.Tensor] = field(default_factory=list)
    whiteboard_state: Optional[torch.Tensor] = None
    _seen_tokens: int = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        return None


# =============================================================================
# 2. RMSNorm & Components
# =============================================================================

class HelixNetRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        return hidden_states * torch.sigmoid(gate)

class HelixNetRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device)
    
    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)
        if position_ids is not None:
            return self.cos_cached[position_ids].to(x.dtype), self.sin_cached[position_ids].to(x.dtype)
        return self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# 3. Hybrid Attention (Token + Whiteboard)
# =============================================================================

class HelixNetHybridAttention(nn.Module):
    """
    A Unified Attention Layer that:
    1. Uses Query (Q) to attend to past tokens (Sliding Window Attention).
    2. Uses a DECOUPLED Query (Q_wb) to read from the Global Whiteboard.
    3. Writes to the Whiteboard using dedicated K_wb and V_wb.
    4. Concatenates [Attn_Output; WB_Output] and projects.
    """
    
    def __init__(self, config: HelixNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        
        # Free Lunch #1: Sliding Window Config
        # Default to 4096 if not in config
        self.sliding_window = getattr(config, "sliding_window", 4096)
        
        # --- 1. Standard Attention Config ---
        self.num_attn_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attn_dropout = config.attention_dropout
        
        # --- 2. Whiteboard Config ---
        self.num_wb_heads = config.state_bank_num_heads 
        self.wb_head_k_dim = config.state_bank_head_dim
        self.wb_head_v_dim = int(config.state_bank_head_dim * config.state_bank_expand_v)

        # --- Projections ---
        
        # Standard Attention Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_attn_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_attn_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_attn_heads * self.head_dim, bias=False)
        
        # Free Lunch #2: Decoupled Whiteboard Query
        # We NO LONGER reuse self.q_proj. We learn a specific query for state retrieval.
        self.wb_q_proj = nn.Linear(self.hidden_size, self.num_wb_heads * self.wb_head_k_dim, bias=False)
        
        # Whiteboard Write Keys/Values
        self.wb_k_proj = nn.Linear(self.hidden_size, self.num_wb_heads * self.wb_head_k_dim, bias=False)
        self.wb_v_proj = nn.Linear(self.hidden_size, self.num_wb_heads * self.wb_head_v_dim, bias=False)
        
        # Whiteboard Gates
        self.wb_beta_proj = nn.Linear(self.hidden_size, self.num_wb_heads, bias=False) 
        self.wb_decay_proj = nn.Linear(self.hidden_size, self.num_wb_heads, bias=True) 
        
        # Initialize Whiteboard Decay Bias (Fast vs Slow heads)
        self._init_decay_bias()

        # Learnable orthogonality bias for WB
        self.wb_depth_k_emb = nn.Parameter(torch.randn(1, 1, self.num_wb_heads, self.wb_head_k_dim) * 0.02)

        # Output Projection
        self.concat_dim = (self.num_attn_heads * self.head_dim) + (self.num_wb_heads * self.wb_head_v_dim)
        self.o_proj = nn.Linear(self.concat_dim, self.hidden_size, bias=False)
        
        # Output Norm for WB part
        self.wb_o_norm = FusedRMSNormGated(self.wb_head_v_dim, eps=config.rms_norm_eps)
        self.wb_g_proj = nn.Linear(self.hidden_size, self.num_wb_heads * self.wb_head_v_dim, bias=False)

        # RoPE
        self.rotary_emb = HelixNetRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def _init_decay_bias(self):
        """
        Initializes decay projection bias to enforce specific time-scale bands.
        Fast Heads (Low Retention) vs Slow Heads (High Retention).
        """
        n_heads = self.num_wb_heads
        half_heads = n_heads // 2
        
        with torch.no_grad():
            fast_retention = torch.rand(half_heads) * (0.1 - 0.01) + 0.01
            slow_retention = torch.rand(n_heads - half_heads) * (0.999 - 0.9) + 0.9
            target_sigmoid = torch.cat([fast_retention, slow_retention])
            bias_init = torch.log(target_sigmoid / (1 - target_sigmoid))
            self.wb_decay_proj.bias.copy_(bias_init)
            self.wb_decay_proj.weight.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        whiteboard_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[WhiteboardCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[WhiteboardCache]]:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # =====================================================================
        # Part A: Standard Attention (Sliding Window)
        # =====================================================================
        
        # 1. Project Q, K, V
        q = self.q_proj(hidden_states)
        k_attn = self.k_proj(hidden_states)
        v_attn = self.v_proj(hidden_states)
        
        q = q.view(batch_size, seq_len, self.num_attn_heads, self.head_dim)
        k_attn = k_attn.view(batch_size, seq_len, self.num_attn_heads, self.head_dim)
        v_attn = v_attn.view(batch_size, seq_len, self.num_attn_heads, self.head_dim)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(v_attn, position_ids)
        q_rope, k_rope = apply_rotary_pos_emb(q, k_attn, cos, sin, position_ids, unsqueeze_dim=2)
        
        # Update Cache
        # Note: If using sliding window, we theoretically could drop old keys from cache to save VRAM.
        # But for simplicity/correctness with standard Cache API, we keep them (cache management is usually external).
        if past_key_value is not None:
            # Transpose to (B, H, T, D) for cache update
            k_update = k_rope.transpose(1, 2)
            v_update = v_attn.transpose(1, 2)
            k_update, v_update = past_key_value.update(k_update, v_update, self.layer_idx)
            
            # For SDPA, we need (B, H, T, D)
            k_sdpa = k_update
            v_sdpa = v_update
            q_sdpa = q_rope.transpose(1, 2)
        else:
            q_sdpa = q_rope.transpose(1, 2)
            k_sdpa = k_rope.transpose(1, 2)
            v_sdpa = v_attn.transpose(1, 2)

        # Free Lunch #1: Efficient Sliding Window
        attn_output = None
        
        # Path 1: Flash Attention (Preferred for Sliding Window)
        if FLASH_ATTN_AVAILABLE:
            # flash_attn expects (B, T, H, D)
            q_flash = q_sdpa.transpose(1, 2)
            k_flash = k_sdpa.transpose(1, 2)
            v_flash = v_sdpa.transpose(1, 2)
            
            # Using causal=True + window_size implies "Causal Sliding Window"
            attn_output = flash_attn_func(
                q_flash, k_flash, v_flash,
                dropout_p=self.attn_dropout if self.training else 0.0,
                softmax_scale=None,
                causal=True,
                window_size=(self.sliding_window, 0) # (left, right)
            )
            # Flatten
            attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Path 2: SDPA Fallback (Manual Masking)
        else:
            # We must construct a sliding window mask
            # Mask shape: (B, 1, T_q, T_k)
            t_q = q_sdpa.size(2)
            t_k = k_sdpa.size(2)
            
            # Base Causal Mask
            # If attention_mask is provided (padding mask), combine it
            if attention_mask is not None:
                # Expand (B, 1, 1, T) -> (B, 1, T_q, T_k) usually handled by broadcasting
                # We assume attention_mask is the standard padding mask
                # SDPA handles broadcasting
                pass
            
            # Sliding Window Logic via Bias or Mask
            # Creating a bias matrix is memory intensive for long sequences, 
            # but standard SDPA is O(N^2) anyway.
            
            # Create indices
            device = q_sdpa.device
            q_idx = torch.arange(t_q, device=device).unsqueeze(1) # (T_q, 1)
            # Adjust k_idx for past cache offset
            past_len = t_k - t_q
            k_idx = torch.arange(t_k, device=device).unsqueeze(0) # (1, T_k)
            
            # Logic: |(q_idx + past_len) - k_idx| <= window
            # And causal: (q_idx + past_len) >= k_idx
            
            # Just create a boolean mask (1 to attend, 0 to mask)
            # causal mask: row >= col
            causal_mask = (q_idx + past_len) >= k_idx
            # window mask: row - col <= window
            window_mask = ((q_idx + past_len) - k_idx) <= self.sliding_window
            
            combined_mask = causal_mask & window_mask
            
            # If padding mask exists
            if attention_mask is not None:
                # attention_mask is usually (B, 1, 1, T_k)
                combined_mask = combined_mask.unsqueeze(0).unsqueeze(0) & (attention_mask > 0)
                
            attn_output = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=combined_mask if not FLASH_ATTN_AVAILABLE else None, # If Flash not available, use mask
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False # We handled causality in the mask
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, -1)

        # =====================================================================
        # Part B: Whiteboard Interaction (Decoupled Query)
        # =====================================================================
        
        # 1. Generate Query for Whiteboard (Free Lunch #2)
        # Use specific projection rather than reusing attn q
        q_wb = self.wb_q_proj(hidden_states)
        q_wb = F.silu(q_wb) 
        
        # 2. Project K, V for Whiteboard Writing
        k_wb = self.wb_k_proj(hidden_states)
        v_wb = self.wb_v_proj(hidden_states)
        k_wb = F.silu(k_wb)
        v_wb = F.silu(v_wb)
        
        # Reshape
        k_wb = rearrange(k_wb, 'b t (h d) -> b t h d', h=self.num_wb_heads)
        v_wb = rearrange(v_wb, 'b t (h d) -> b t h d', h=self.num_wb_heads)
        
        # Add orthogonality bias
        k_wb = k_wb + self.wb_depth_k_emb.to(k_wb.dtype)
        
        # 3. Gates
        beta = torch.sigmoid(self.wb_beta_proj(hidden_states)) 
        g_log = F.logsigmoid(self.wb_decay_proj(hidden_states)) 
        
        # 4. Delta Rule (Read + Update)
        wb_read_out, next_whiteboard_state = chunk_gated_delta_rule(
            q=q_wb,
            k=k_wb,
            v=v_wb,
            g=g_log,
            beta=beta,
            scale=self.wb_head_k_dim ** -0.5,
            initial_state=whiteboard_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True
        )
        
        # 5. Output Gating/Norm for Whiteboard
        wb_g_out = rearrange(self.wb_g_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_wb_heads)
        wb_read_out = self.wb_o_norm(wb_read_out, wb_g_out)
        
        # Flatten WB output
        wb_read_out = rearrange(wb_read_out, 'b t h d -> b t (h d)')
        
        # =====================================================================
        # Part C: Fusion
        # =====================================================================
        
        fused_output = torch.cat([attn_output, wb_read_out], dim=-1)
        final_output = self.o_proj(fused_output)
        
        return final_output, next_whiteboard_state, None, past_key_value


# =============================================================================
# 4. MLP & Decoder Layer (Unchanged)
# =============================================================================

class HelixNetMLP(nn.Module):
    def __init__(self, config: HelixNetConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = F.silu
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class HelixNetDecoderLayer(nn.Module):
    def __init__(self, config: HelixNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = HelixNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HelixNetHybridAttention(config, layer_idx)
        self.post_attention_layernorm = HelixNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HelixNetMLP(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        whiteboard_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[WhiteboardCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output, new_whiteboard_state, attn_weights, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            whiteboard_state=whiteboard_state,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states, new_whiteboard_state, attn_weights


# =============================================================================
# 5. Models (Base, Config, LM Head)
# =============================================================================

class HelixNetPreTrainedModel(PreTrainedModel):
    config_class = HelixNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HelixNetDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "whiteboard_state"]
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class HelixNetModel(HelixNetPreTrainedModel):
    def __init__(self, config: HelixNetConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([HelixNetDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = HelixNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Params for State Initialization
        self.wb_num_heads = config.state_bank_num_heads
        self.wb_head_k_dim = config.state_bank_head_dim
        self.wb_head_v_dim = int(config.state_bank_head_dim * config.state_bank_expand_v)
        
        # Free Lunch #3: Learnable Aggregation Weights
        # Initialize to uniform (all layers equal) but learnable
        self.layer_agg_weights = nn.Parameter(torch.zeros(config.num_hidden_layers))
        
        self.post_init()
    
    def get_input_embeddings(self): return self.embed_tokens
    def set_input_embeddings(self, value): self.embed_tokens = value
    
    def _init_whiteboard_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.wb_num_heads, self.wb_head_k_dim, self.wb_head_v_dim, device=device, dtype=dtype)
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        batch_size, seq_length, _ = hidden_states.shape
        
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values else 0
            position_ids = torch.arange(past_seen, past_seen + seq_length, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            
        if use_cache and past_key_values is None:
            past_key_values = WhiteboardCache()
            
        # 1. Init Whiteboard
        if past_key_values is not None and past_key_values.whiteboard_state is not None:
            initial_whiteboard_state = past_key_values.whiteboard_state
        else:
            initial_whiteboard_state = self._init_whiteboard_state(batch_size, hidden_states.device, hidden_states.dtype)
            
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        layer_state_outputs = []
        
        # 2. Forward Layers
        for decoder_layer in self.layers:
            if output_hidden_states: all_hidden_states += (hidden_states,)
            
            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                whiteboard_state=initial_whiteboard_state, 
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            layer_state_outputs.append(layer_outputs[1]) # Collect proposed S_t
            if output_attentions: all_self_attns += (layer_outputs[2],)
            
        hidden_states = self.norm(hidden_states)
        if output_hidden_states: all_hidden_states += (hidden_states,)
        
        # 3. Aggregate Whiteboard Updates (Free Lunch #3: Weighted Sum)
        if len(layer_state_outputs) > 0:
            # Stack: (Num_Layers, B, H, K, V)
            stacked_states = torch.stack(layer_state_outputs, dim=0)
            
            # Softmax the learnable weights
            agg_probs = F.softmax(self.layer_agg_weights, dim=0) # (Num_Layers,)
            
            # Reshape for broadcast: (Num_Layers, 1, 1, 1, 1)
            agg_probs = agg_probs.view(-1, 1, 1, 1, 1)
            
            # Weighted Sum
            current_whiteboard_state = torch.sum(stacked_states * agg_probs, dim=0)
        else:
            current_whiteboard_state = initial_whiteboard_state
            
        if use_cache and past_key_values is not None:
            past_key_values.whiteboard_state = current_whiteboard_state.detach()
            
        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class HelixNetForCausalLM(HelixNetPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config: HelixNetConfig):
        super().__init__(config)
        self.model = HelixNetModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        
    def get_input_embeddings(self): return self.model.embed_tokens
    def set_input_embeddings(self, value): self.model.embed_tokens = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = nn.CrossEntropyLoss()(shift_logits, shift_labels)
            
        if not return_dict:
            return ((loss,) + (logits,) + outputs[1:]) if loss is not None else ((logits,) + outputs[1:])
            
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values is not None:
            cache_len = past_key_values.get_seq_length()
            if cache_len > 0: input_ids = input_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None: position_ids = position_ids[:, -input_ids.shape[1]:]
            
        model_inputs = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None and past_key_values is None else {"input_ids": input_ids}
        model_inputs.update({"position_ids": position_ids, "past_key_values": past_key_values, "use_cache": kwargs.get("use_cache"), "attention_mask": attention_mask})
        return model_inputs
