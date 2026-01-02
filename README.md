# Stereo DDP + TOML + TensorBoard (Template)

This is a minimal, *runnable* template for training a stereo model with PyTorch that supports:
- **Single-node multi-GPU** distributed training via `torchrun`
- Configs in **TOML** (namespaced by function: `[train]`, `[data]`, `[model]`, `[loss]`)
- **TensorBoard** logging of scalars and images (left/right, predicted disparity, GT disparity)
- Saving a copy of the **effective config** into the run directory

> The dataset and model are **dummy** but wired like a real project so you can replace them.

## Quick Start

```bash
# 1) Create/activate your env; install deps
pip install torch torchvision tensorboard toml tomli_w scipy pillow

# 2) Launch TensorBoard (optional)
tensorboard --logdir outputs

# 3) Run on 4 GPUs (single node)
torchrun --nproc_per_node=4 --master_port=29501 train_stereo_la.py \
  --config configs/example_config.toml --launcher pytorch
```

- Logs/checkpoints go to: `outputs/run_debug/`
- TensorBoard logs: `outputs/run_debug/tb/`
- Effective config copy: `outputs/run_debug/config_used.toml`

## Config (TOML)

`configs/example_config.toml` illustrates a clean namespaced layout:

```toml
[train]
output_dir = "outputs/run_debug"
epochs = 3
batch_size = 4
lr = 1e-3
weight_decay = 1e-4
amp = true
grad_clip = 1.0
seed = 42
log_interval = 10

[data]
train_length = 128
val_length = 32
height = 192
width = 320
num_workers = 2
max_disp = 64

[model]
features = 32
predict_pose = true

[loss]
w_disp = 1.0
w_pose = 0.1
```

> The script will **read** TOML using `tomllib` (Py3.11+) or fall back to `toml`.  
> It will **write** TOML using `tomli_w` if available; otherwise it copies the original text and appends a `[runtime]` block.

## What gets logged to TensorBoard?

- Scalars:
  - `train/loss`, `train/loss_disp`, `train/loss_pose`
  - `val/disp_l1`
  - `val/pose_mse` (if pose prediction enabled)
- Images (grids):
  - `train/left_right_disp_dispgt` (left, right, predicted disp, GT disp)
  - `val/left_right_disp_dispgt`

## Replace the Dummy Parts

- **Dataset**: `data/dummy_stereo_dataset.py`
  - Replace with your own stereo dataset (load left/right images and GT disparity; optional pose labels)
- **Model**: `models/simple_stereo.py`
  - Replace with your own architecture (keep output dict keys: `"disp"` and optional `"pose"`)

## Notes

- Uses `DistributedSampler` for both train/val.
- Initializes DDP automatically when launched with `torchrun` (env://).
- Saves checkpoints as `checkpoint_000.pth`, etc. (rank-0 only).
