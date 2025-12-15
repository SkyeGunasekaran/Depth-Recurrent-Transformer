# Depth-Gated DeltaNet

A hybrid transformer architecture combining standard attention with **depth-recurrent state banks** using the Gated Delta Rule. The state matrix evolves across layers (depth) rather than across time steps, enabling deeper effective computation and long-term information persistence.

## Key Innovation

Traditional transformers process information layer by layer with no explicit state passing between layers. **Depth-Gated DeltaNet** introduces a persistent state matrix $S$ that:

1. **Flows across layers**: Layer $L$ receives state from layer $L-1$, updates it, and passes it to layer $L+1$
2. **Uses the Gated Delta Rule**: State updates follow $S_{t+1} = \alpha_t \cdot S_t + \beta_t \cdot (v_t - S_t k_t) k_t^T$
3. **Provides queryable memory**: Each layer can query the accumulated state to retrieve relevant information

```
Layer 0: x → [Attention + MLP] → [State Bank] → y₀, S₀
                                      ↓
Layer 1: y₀ → [Attention + MLP] → [State Bank(S₀)] → y₁, S₁
                                      ↓
Layer 2: y₁ → [Attention + MLP] → [State Bank(S₁)] → y₂, S₂
                                      ↓
                                     ...
```

## Architecture

### HybridDepthBlock

Each layer consists of:

1. **Main Stream** (Standard Llama-style):
   - Pre-norm → Multi-Head Attention → Residual
   - Pre-norm → SwiGLU MLP → Residual

2. **State Bank** (Depth-Recurrent):
   - Pre-norm → Gated DeltaNet → State Injection
   - Receives state from previous layer
   - Returns updated state for next layer

3. **State Injection**:
   - Residual: `output = main_stream + scale * state_query`
   - Gated: `output = main_stream + gate * state_query`

### Gated Delta Rule

The state bank uses the Gated Delta Rule for updates:

```
α_t = exp(g_t)                    # Decay/retention gate
S_{t+1} = α_t · S_t + β_t · δ_t   # State update
δ_t = (v_t - S_t · k_t) ⊗ k_t     # Delta term
o_t = S_t · q_t                   # Query output
```

Where:
- `α_t`: Controls how much previous state is retained
- `β_t`: Controls write strength of new information
- `δ_t`: The "delta" - difference between new value and what state would predict

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/depth-deltanet.git
cd depth-deltanet

# Install dependencies
pip install torch transformers einops

# Optional: Install flash-linear-attention for optimized kernels
pip install fla
```

## Quick Start

### Basic Usage

```python
from depth_deltanet import DepthDeltaNetConfig, DepthDeltaNetForCausalLM

# Create configuration
config = DepthDeltaNetConfig(
    vocab_size=32000,
    hidden_size=2048,
    num_hidden_layers=24,
    num_attention_heads=32,
    # State bank configuration
    state_bank_num_heads=16,
    state_bank_head_dim=64,
    state_bank_expand_v=2.0,
    depth_init=True,  # Crucial for layer-to-layer persistence
)

# Create model
model = DepthDeltaNetForCausalLM(config)

# Forward pass
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
```

### Generation with Depth Caching

```python
# The cache stores both KV cache AND depth states
outputs = model.generate(
    input_ids=prompt_ids,
    max_new_tokens=100,
    use_cache=True,  # Enables depth state caching
    do_sample=True,
    temperature=0.8,
)
```

### Using Predefined Configurations

```python
from depth_deltanet.auto_registration import get_config_for_size

# Available sizes: tiny, small, medium, large, xl, 3b, 7b
config = get_config_for_size("medium")
model = DepthDeltaNetForCausalLM(config)
```

## Configuration Options

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 2048 | Hidden dimension |
| `num_hidden_layers` | 24 | Number of transformer layers |
| `num_attention_heads` | 32 | Attention heads |
| `head_dim` | 64 | Dimension per head |
| `intermediate_size` | auto | MLP intermediate dimension |

### State Bank

| Parameter | Default | Description |
|-----------|---------|-------------|
| `state_bank_num_heads` | same as attn | Number of state bank heads |
| `state_bank_head_dim` | same as attn | State bank head dimension |
| `state_bank_expand_v` | 2.0 | Value expansion factor |
| `state_bank_use_short_conv` | True | Use short conv for local mixing |
| `depth_init` | True | Initialize for state persistence |
| `state_injection_mode` | "residual" | How to inject state output |

### Training Considerations

The `depth_init=True` setting is **crucial** for depth recurrence to work effectively:
- Initializes decay parameters (`dt_bias`) to be small
- Results in `α ≈ 1.0` (high retention)
- Allows state to persist meaningfully across layers

## Training

### Basic Training Loop

```python
from depth_deltanet.training_utils import (
    TrainingConfig,
    create_optimizer_and_scheduler,
)

train_config = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=0.1,
    warmup_steps=1000,
    num_training_steps=100000,
    # Special handling for depth-recurrent params
    decay_param_lr_multiplier=0.1,  # Lower LR for A_log, dt_bias
)

optimizer, scheduler = create_optimizer_and_scheduler(model, train_config)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### HuggingFace Trainer Integration

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=1000,
    num_train_epochs=3,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

## Caching Mechanism

The `DepthDeltaNetCache` manages two types of state:

1. **KV Cache**: Standard attention key-value cache for efficient autoregressive generation
2. **Depth State**: The state matrix from each layer's state bank

During generation:
- First token: Full forward pass, cache KV and depth states
- Subsequent tokens: Single token forward, reuse cached states

```python
# Manual cache management
cache = DepthDeltaNetCache()

# First forward pass
outputs = model(prompt_ids, use_cache=True)
cache = outputs.past_key_values

# Incremental decoding
for _ in range(num_tokens):
    outputs = model(next_token, past_key_values=cache, use_cache=True)
    cache = outputs.past_key_values
    next_token = sample(outputs.logits[:, -1])
```

## Model Sizes

| Size | Params | Hidden | Layers | Heads | State Bank Heads |
|------|--------|--------|--------|-------|------------------|
| tiny | ~25M | 512 | 8 | 8 | 4 |
| small | ~125M | 768 | 12 | 12 | 8 |
| medium | ~350M | 1024 | 24 | 16 | 8 |
| large | ~760M | 1536 | 24 | 16 | 12 |
| xl | ~1.3B | 2048 | 24 | 16 | 16 |
| 3b | ~3B | 2560 | 32 | 32 | 16 |
| 7b | ~7B | 4096 | 32 | 32 | 16 |

## Integration with HuggingFace

### Auto Registration

```python
from depth_deltanet.auto_registration import register_auto_classes

register_auto_classes()

# Now you can use Auto classes
from transformers import AutoModelForCausalLM, AutoConfig

model = AutoModelForCausalLM.from_pretrained("path/to/model")
```

### Saving with Auto Map

```python
from depth_deltanet.auto_registration import save_pretrained_with_auto_map

# Saves model with auto_map for standalone loading
save_pretrained_with_auto_map(model, "path/to/save")
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_model.py::TestCaching -v
pytest tests/test_model.py::TestGradients -v
```

## Technical Details

### Why Depth Recurrence?

Standard transformers have no explicit communication between layers except through the residual stream. Each layer independently processes the same representation. Depth recurrence provides:

1. **Explicit State Accumulation**: Information can be explicitly stored and retrieved
2. **Layer Specialization**: Earlier layers can store, later layers can retrieve
3. **Compute Amortization**: Heavy computations can be cached in state
4. **Deeper Effective Depth**: State provides a "shortcut" for information flow

### Stability Considerations

Depth recurrence requires careful initialization:

1. **Decay Initialization**: `depth_init=True` ensures high retention (`α ≈ 1`)
2. **Gradient Clipping**: Important due to multiplicative state updates
3. **Lower LR for Decay Params**: `A_log` and `dt_bias` need gentle training

### Computational Cost

The state bank adds overhead per layer:
- **Memory**: `O(batch × heads × head_dim² × expand_v)` per layer
- **Compute**: Similar to attention for state update, less for query

For long sequences, the state bank can be more efficient than full attention since state size is fixed regardless of sequence length.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Gated Delta Rule from [Flash Linear Attention](https://github.com/sustcsonglin/flash-linear-attention)
- Architecture inspired by Llama-2 and Mamba
