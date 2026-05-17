# LoRA in Stable Audio 3

LoRA (Low-Rank Adaptation) enables parameter-efficient fine-tuning of large diffusion models. Instead of updating all model weights during training, LoRA freezes the pre-trained weights and injects small, trainable low-rank matrices into each layer. This dramatically reduces the number of trainable parameters and the size of saved checkpoints while still allowing the model to learn new behaviors.

Stable Audio 3 supports a family of adapter types that trade off expressiveness against parameter count:

| Adapter | Trainable Params per Layer | Use Case |
|---------|--------------------------|----------|
| **lora** | `rank * (fan_in + fan_out)` | General-purpose fine-tuning. Good balance of expressiveness and efficiency. |
| **dora-rows** (default) | `rank * (fan_in + fan_out) + fan_out` | DoRA with per-row (per-output-neuron) magnitude. Paper-correct variant. |
| **dora-cols** | `rank * (fan_in + fan_out) + fan_in` | DoRA with per-column (per-input-feature) magnitude. |
| **bora** | `rank * (fan_in + fan_out) + fan_in + fan_out` | Bi-dimensional DoRA — independent row and column magnitudes. |
| **lora-xs** | `rank²` | Maximum parameter efficiency. Only a tiny core matrix is trainable; bases are frozen SVD factors. |
| **dora-rows-xs** | `rank² + fan_out` | DoRA-rows + LoRA-XS. Frozen SVD bases plus per-row magnitude. |
| **dora-cols-xs** | `rank² + fan_in` | DoRA-cols + LoRA-XS. Frozen SVD bases plus per-column magnitude. |
| **bora-xs** | `rank² + fan_in + fan_out` | BoRA + LoRA-XS. Frozen SVD bases plus both row and column magnitudes. |

LoRA is applied to both the diffusion backbone (DiT) and the conditioner (only trainable parameters like the `seconds_total` conditioner).

# Adapter Types

## LoRA (Standard)

Standard LoRA decomposes each weight update into two low-rank matrices:

```
W' = W + (alpha/rank) * B @ A
```

Where `W` is the frozen pre-trained weight, `A` is a `(rank, fan_in)` matrix, and `B` is a `(fan_out, rank)` matrix. Only `A` and `B` are trained. The `alpha/rank` ratio controls the scaling of the LoRA update relative to the original weight.

`A` is initialized with Kaiming uniform initialization and `B` is initialized to zeros, so the LoRA update starts at zero and the model begins training from its pre-trained behavior.

## DoRA (default: dora-rows)

DoRA (Weight-Decomposed Low-Rank Adaptation) extends standard LoRA by separating weight updates into direction and magnitude components:

```
V = W + (alpha/rank) * B @ A       (low-rank update, same as LoRA)
V_hat = V / ||V||_row               (row-normalized direction)
W' = V_hat * magnitude              (scale by per-row magnitude)
```

The `magnitude` vector is initialized from the row norms of the original pre-trained weight: `||W||_row`. This means the model starts with the same effective weights as the pre-trained model, but can independently adjust the direction and magnitude of each row during training.

There are two variants:
- **`dora-rows`** (default): magnitude per output neuron (`fan_out` values). This is the paper-correct variant.
- **`dora-cols`**: magnitude per input feature (`fan_in` values).

## BoRA

BoRA (Bi-dimensional DoRA) applies independent magnitude scaling to both rows and columns:

```
V = W + (alpha/rank) * B @ A
V_r = V / ||V||_row                 (row-normalize)
H_r = magnitude_r * V_r             (scale rows)
H_c = H_r / ||H_r||_col            (column-normalize)
W' = H_c * magnitude_c              (scale columns)
```

This adds both a `magnitude_r` vector of size `fan_out` and a `magnitude_c` vector of size `fan_in`. More expressive than either DoRA variant, at the cost of slightly more parameters.

## LoRA-XS

LoRA-XS takes parameter efficiency to the extreme by freezing the low-rank bases and only training a tiny core matrix:

```
U, S, V^T = SVD(W)                  (SVD of pre-trained weight)
W' = W + (alpha/rank) * U[:, :r] @ M_xs @ V[:, :r]^T
```

`U` and `V` are frozen buffers derived from the SVD of the original weight. Only `M_xs`, a `(rank, rank)` matrix, is trainable. For a typical rank of 8, this means only 64 trainable parameters per layer, compared to thousands for standard LoRA.

Because LoRA-XS computes SVD during initialization, placing the model on GPU before adding LoRA enables GPU-accelerated SVD, which is significantly faster for large models. The inference code does this automatically.

For training, computing SVD for every layer at startup can be slow. Use `--svd_bases_path` to pass a pre-computed bases file instead (see [Training Configuration](#training-configuration)).

## -XS Hybrid Variants

The `-xs` suffix can be combined with any DoRA or BoRA variant:

- **`dora-rows-xs`**: Frozen SVD bases + `M_xs` core + per-row magnitude
- **`dora-cols-xs`**: Frozen SVD bases + `M_xs` core + per-column magnitude
- **`bora-xs`**: Frozen SVD bases + `M_xs` core + both row and column magnitudes

These combine the parameter efficiency of LoRA-XS with the magnitude decomposition of DoRA/BoRA.

# Training Configuration

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--rank` | 16 | LoRA rank. Lower = fewer parameters, higher = more expressive. |
| `--lora_alpha` | same as `--rank` | Scaling factor. The effective scaling is `alpha / rank`. Setting `alpha = rank` gives a scaling of 1.0. |
| `--adapter_type` | `dora` | Adapter type. See table above for all options. |
| `--dropout` | 0.0 | Dropout probability applied to LoRA inputs during training. |
| `--include` | (all layers) | Only apply LoRA to modules whose name contains one of these substrings. |
| `--exclude` | (no exclusions) | Skip modules matching any of these substrings, even if they match `--include`. |
| `--svd_bases_path` | None | Path to a pre-computed SVD bases `.pt` file. Used by `-XS` adapter types to skip per-layer SVD at startup. |
| `--base_precision` | None | Cast frozen base weights to `bf16`, `bfloat16`, `fp16`, or `float16` after applying LoRA. LoRA params remain in fp32. Reduces VRAM usage. |
| `--lora_checkpoint` | None | Path to an existing `.safetensors` LoRA checkpoint to resume training from. |

## Layer Filtering

You can control which layers receive LoRA using `--include` and `--exclude`. Both accept lists of substring patterns with bracket expansion support (e.g., `layers[0-11]`).

When `--include` is specified, only layers whose fully-qualified name contains at least one of the include substrings are candidates for LoRA. When `--exclude` is specified, any layer matching an exclude pattern is skipped, even if it also matches an include pattern. When neither is specified, all layers of matching types (Linear, Conv1d) receive LoRA.

### Examples

Exclude the `seconds_total` conditioner (prevents conditioner hijacking on small datasets):
```bash
uv run python scripts/train_lora.py --model medium-base --data_dir ./my_data \
    --rank 16 --adapter_type lora-xs --exclude seconds_total
```

Only apply LoRA to transformer layers:
```bash
uv run python scripts/train_lora.py --model medium-base --data_dir ./my_data \
    --include transformer.layers
```

Only the first 12 transformer layers:
```bash
uv run python scripts/train_lora.py --model medium-base --data_dir ./my_data \
    --include "layers[0-11]"
```

Everything except local embedding and seconds_total conditioner:
```bash
uv run python scripts/train_lora.py --model medium-base --data_dir ./my_data \
    --exclude to_local_embed seconds_total
```

Layer names are matched against the module's fully-qualified name relative to the submodel root. For the diffusion backbone these look like `transformer.layers.0.self_attn.to_qkv`, `to_timestep_embed.0`, etc. For the conditioner: `conditioners.seconds_total.embedder.embedding.1`, etc.

The `include`/`exclude` config is persisted in the checkpoint and automatically applied when loading the LoRA at inference time.

## Memory Optimisation

**`--base_precision bf16`** casts all frozen base model weights to bfloat16 after LoRA is applied, while keeping LoRA parameters in fp32 for the optimizer. This halves the VRAM used by the frozen weights with negligible quality impact on most tasks:

```bash
uv run python scripts/train_lora.py --model medium-base --data_dir ./my_data \
    --adapter_type dora-rows --base_precision bf16
```

**Pre-computed SVD bases** avoid the per-layer SVD computation at startup for `-XS` adapters. Compute once and reuse across training runs:

```bash
uv run python scripts/train_lora.py --model medium-base --data_dir ./my_data \
    --adapter_type lora-xs --svd_bases_path ./svd_bases.pt
```

Without `--svd_bases_path`, SVD is computed on-the-fly per layer (slower startup, same training quality).

## Resuming Training

Use `--lora_checkpoint` to continue training from an existing checkpoint. The LoRA weights are loaded with `strict=False`, so the checkpoint does not need to cover exactly the same set of layers:

```bash
uv run python scripts/train_lora.py --model medium-base --data_dir ./my_data \
    --lora_checkpoint ./lora_out/lora_step500.safetensors \
    --steps 1000 --output_dir ./lora_out_continued
```

# How LoRA Training Works

When LoRA training is run:

1. **Base weights are frozen.** Both the diffusion model and conditioner are set to `eval()` mode with `requires_grad_(False)`. No base model weights are updated during training.

2. **SVD bases are loaded (for -XS adapters).** If `--svd_bases_path` is provided, pre-computed `U`/`V` bases are loaded from disk. Otherwise, SVD is computed per layer at initialisation time.

3. **LoRA layers are added.** `add_lora()` uses PyTorch's `nn.utils.parametrize` API to add LoRA parametrizations to `Linear` and `Conv1d` layers in both the model and conditioner. If `--include` or `--exclude` filters are specified, only matching layers receive LoRA.

4. **Checkpoint is loaded (if resuming).** The LoRA state dict is loaded with `strict=False`.

5. **LoRA params are cast to fp32.** All LoRA parameters are moved to float32 so the optimizer has full precision, even if the base model is in bfloat16.

6. **Base weights are optionally downcast.** If `--base_precision` is set, all frozen base weights (including the pretransform) are cast to bf16 or fp16 to reduce VRAM usage.

7. **Only LoRA params are optimised.** The optimizer receives only the LoRA parameters (collected via `get_lora_params()`), not the full model parameters.

8. **Checkpoints save only LoRA.** On save, `get_lora_state_dict()` extracts only the LoRA tensors and saves them as `.safetensors` with the `lora_config` embedded in the file metadata.

# Inference with LoRA

## Loading LoRA Checkpoints

Use the `--lora-ckpt-path` argument when running the Gradio interface. You can load one or multiple LoRAs:

```bash
# Single LoRA
uv run python run_gradio.py --model medium-base --lora-ckpt-path lora.safetensors

# Multiple LoRAs
uv run python run_gradio.py --model medium-base \
    --lora-ckpt-path style_a.safetensors style_b.safetensors
```

The loading process (`load_and_apply_loras`):
1. The base model is created and loaded normally
2. The model is moved to GPU (important for LoRA-XS GPU-accelerated SVD)
3. SVD bases are loaded once if any LoRA is a `-XS` type
4. LoRA parametrizations are added based on the config embedded in each checkpoint
5. LoRA weights are loaded with `strict=False`

When multiple LoRAs are loaded, they are stacked using PyTorch's native `nn.utils.parametrize` API. Each call to `register_parametrization` appends to the `ParametrizationList` on each weight, and the forward chains them additively. Each LoRA is assigned a unique `lora_index` (0, 1, 2, ...) that enables independent control.

Multiple LoRAs can use different adapter types (e.g., one standard LoRA and one DoRA), different ranks, and different layer filters. They are all applied simultaneously during inference.

## Gradio UI Controls

When LoRA checkpoints are loaded, the Gradio interface shows per-LoRA controls. Each LoRA gets its own collapsible accordion with independent settings:

### LoRA DiT Strength
Controls the strength of the LoRA effect on the diffusion model backbone. Default is 1.0 (full effect). Setting to 0 disables the LoRA entirely; values above 1.0 amplify the effect. Range: 0.0 - 10.0.

### LoRA Conditioner Strength
Controls the LoRA effect on the text conditioner independently from the DiT. Default is 1.0. This allows you to, for example, keep the LoRA effect on the conditioner while reducing it on the backbone.

### LoRA Interval
Controls when LoRA is active during the sampling process based on the noise level (sigma). The interval is specified as `[min, max]` where both are in the range 0.0 to 1.0.

- `[0.0, 1.0]` (default): LoRA is active at all noise levels
- `[0.0, 0.95]`: LoRA is active only during the low-noise (late) part of sampling
- `[0.95, 1.0]`: LoRA is active only during the high-noise (early) part of sampling

This can be useful for controlling which aspects of generation the LoRA influences - early steps affect global structure while later steps affect fine details.

### LoRA Layer Filter
A text field that selectively disables LoRA on specific layers by name. Comma-separated substrings are matched against layer names (logical OR). Bracket notation supports ranges.

Examples:
- `.to_global_embed` - disables LoRA on the global embedding projection
- `.transformer.layers[0-5]` - disables LoRA on transformer layers 0 through 5
- `.transformer.layers[0-10], .to_global_embed` - disables both

Layers matching any filter substring are disabled; all other LoRA layers remain active.

### Multi-LoRA Example

With two LoRAs loaded, you could configure them independently:

- **LoRA 1 (style)**: DiT strength 1.0, interval `[0.0, 1.0]` (active everywhere)
- **LoRA 2 (detail)**: DiT strength 0.5, interval `[0.0, 0.5]` (active only in later denoising steps)

Each LoRA's interval and layer filter are evaluated independently at each sampling step. A LoRA is enabled for a step only if the current sigma falls within its interval.

# Advanced Features

## Strength Control

The `set_lora_strength()` function adjusts the LoRA contribution at runtime without modifying the LoRA weights:

```python
from stable_audio_3.models.lora import set_lora_strength

set_lora_strength(model, 0.5)   # Half-strength on all LoRAs
set_lora_strength(model, 0.0)   # Effectively disable all LoRAs
set_lora_strength(model, 2.0)   # Double-strength on all LoRAs

# With multiple LoRAs, target a specific one by index:
set_lora_strength(model, 1.0, lora_index=0)  # Full strength on first LoRA
set_lora_strength(model, 0.5, lora_index=1)  # Half strength on second LoRA
```

This works by scaling the `lora_strength` buffer in each LoRA layer, which multiplies the low-rank delta before adding it to the base weight. When `lora_index` is `None` (the default), all LoRAs are affected.

## Multiple LoRA Merging

You can merge multiple LoRA checkpoints with different weights into a single base model:

```python
from stable_audio_3.models.lora.utils import merge_loras_into_base_model

lora_configurations = [
    {
        'name': 'style_a',
        'state_dict': lora_sd_a,
        'application_weight': 0.7
    },
    {
        'name': 'style_b',
        'state_dict': lora_sd_b,
        'application_weight': 0.3
    }
]

merge_loras_into_base_model(model, lora_configurations)
```

This computes the weighted sum of LoRA deltas and applies them directly to the base weights. After merging, the LoRA parametrizations are disabled (the effect is baked into the base weights). This works with all adapter types (LoRA, DoRA, BoRA, LoRA-XS, and all hybrid variants).

## Weight Tying

For models where the input embedding and output projection share weights, LoRA supports weight tying:

```python
from stable_audio_3.models.lora.utils import tie_weights, untie_weights

tie_weights(linear_layer, embedding_layer)   # Share LoRA params
untie_weights(linear_layer, embedding_layer) # Create independent copies
```

This is only supported for standard LoRA (not DoRA, BoRA, or -XS variants).
