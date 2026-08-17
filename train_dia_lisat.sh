#!/usr/bin/env bash
# DIA-LISAt training example.
#
# Everything before the "DIA" block is the plain LISAt stage-2 recipe; adapt the
# paths / dataset mixture to your setup. The DIA flags all have defaults, so the
# block below is only there to make the choices explicit and easy to ablate.
set -euo pipefail

deepspeed --master_port=24999 train_lisat.py \
  --version="./model/LISAT_PRE-7b-local-remoteclip" \
  --vision-tower="./model/remote_clip_vit_l_14" \
  --vision_pretrained="./sam_vit_h_4b8939.pth" \
  --dataset_dir="./dataset" \
  --dataset="geo_reason_seg||reason_seg||refer_seg" \
  --sample_rates="12,1,3" \
  --geo_reason_seg_data="GeoReasonSeg|train" \
  --exp_name="dia_lisat" \
  --epochs=10 \
  --steps_per_epoch=500 \
  --batch_size=2 \
  --grad_accumulation_steps=4 \
  --lr=3e-4 \
  --precision="bf16" \
  --lora_r=8 \
  --train_mask_decoder \
  \
  `# ---------------- DIA ----------------` \
  --dia_attn_loss_weight=0.1 \
  --dia_attn_loss_mode="mass" \
  --dia_max_delta_ratio=0.5 \
  --dia_num_heads=8 \
  --dia_embed_dim=256 \
  --con_style="clause"

# Ablations (see README section 5):
#   --dia_attn_loss_weight=0     # adapter without alignment supervision
#   --con_style=adjacent         # [CON][SEG] side by side
#   --no_dia_use_dense_pe        # drop SAM positional encoding in the adapter
