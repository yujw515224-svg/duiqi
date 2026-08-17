#!/usr/bin/env bash
# DIA-LISAt training on the LISAt data (GRES / GeoReasonSeg).
#
# Memory profile: these settings (ZeRO-3, batch 1, grad_accum 8, max_length 512)
# are what fits on 2x24GB (RTX 4090). On 8x A100-80GB you can go back to
# --zero_stage 2 --batch_size 2 --model_max_length 1024, which is closer to the
# LISAt paper recipe.
#
# Expected layout under --dataset_dir:
#   dataset/reason_seg/GeoReasonSeg/{train,val,test}/*.json   (+ the .jpg they name)
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-lisa}"
fi
cd "$(dirname "$0")"

export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

deepspeed --master_port "${MASTER_PORT:-16140}" train_lisat.py \
  --version="./model/LISAT_PRE-7b-local-remoteclip" \
  --vision-tower="./model/remote_clip_vit_l_14" \
  --vision_pretrained="./sam_vit_h_4b8939.pth" \
  --dataset_dir="./dataset" \
  --dataset="geo_reason_seg" \
  --sample_rates="1" \
  --geo_reason_seg_data="GeoReasonSeg|train" \
  --eval_dataset="geo_reason_seg" \
  --eval_split="val" \
  --best_metric="val_giou" \
  --exp_name="${EXP_NAME:-dia_lisat}" \
  --epochs=10 \
  --steps_per_epoch=500 \
  --batch_size=1 \
  --grad_accumulation_steps=8 \
  --zero_stage=3 \
  --zero_bucket_size=2e8 \
  --model_max_length=512 \
  --num_classes_per_sample=1 \
  --workers=4 \
  --lr=3e-4 \
  --precision="bf16" \
  --lora_r=8 \
  --train_mask_decoder \
  --no_auto_resume \
  \
  `# ---------------- DIA ----------------` \
  --dia_attn_loss_weight=0.1 \
  --dia_attn_loss_mode="mass" \
  --dia_max_delta_ratio=0.5 \
  --dia_num_heads=8 \
  --dia_embed_dim=256 \
  --con_style="clause"

# Baseline for the ablation table (same data, schedule and eval path):
#   EXP_NAME=baseline_lisat bash train_dia_lisat.sh --baseline_lisat
#   ...or copy this file and add --baseline_lisat.
#
# 20-step smoke test before committing GPU hours:
#   EXP_NAME=dia_smoke bash train_dia_lisat.sh   # then Ctrl-C after the first
#   prints, or append: --epochs=1 --steps_per_epoch=20 --print_freq=1 --eval_samples=8
#
# Other ablations (docs/DIA_README.md section 4):
#   --dia_attn_loss_weight=0     # adapter without alignment supervision
#   --con_style=none             # no [CON]: concept query falls back to [SEG]
#   --con_style=adjacent         # [CON][SEG] side by side
#   --no_dia_use_dense_pe        # drop SAM positional encoding in the adapter
#
# Mixing in the natural-image sets the LISAt paper also used (needs RefCOCO /
# ReasonSeg data and scikit-image):
#   --dataset="geo_reason_seg||reason_seg||refer_seg" --sample_rates="12,1,3"
