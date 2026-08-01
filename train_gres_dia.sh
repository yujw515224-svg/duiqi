#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT=/root/miniconda3
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate lisa

cd "$(dirname "$0")"

export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
MASTER_PORT="${MASTER_PORT:-16140}"

# DIA-LISAt on GRES / GeoReasonSeg. The batch/ZeRO settings mirror the safe
# baseline run on 2x RTX 4090, while keeping outputs in a separate DIA folder.
deepspeed --master_port "${MASTER_PORT}" train_lisat.py \
  --version ./model/LISAt-7b-local-remoteclip \
  --vision-tower ./model/remote_clip_vit_l_14 \
  --vision_pretrained ./sam_vit_h_4b8939.pth \
  --dataset_dir ./dataset \
  --dataset geo_reason_seg \
  --sample_rates 1 \
  --geo_reason_seg_data "GeoReasonSeg|train" \
  --eval_dataset geo_reason_seg \
  --eval_split val \
  --best_metric val_giou \
  --exp_name dia_lisat_gres_speedmem \
  --model_max_length 512 \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --zero_stage 3 \
  --zero_bucket_size 2e8 \
  --epochs 10 \
  --steps_per_epoch 500 \
  --workers 4 \
  --num_classes_per_sample 1 \
  --lr 3e-4 \
  --print_freq 10 \
  --save_visualizations \
  --vis_samples 16 \
  --attn_loss_weight 0.1 \
  --dia_num_heads 8 \
  --dia_num_evidence_tokens 4
