#!/bin/bash
set -euo pipefail

# Aggressive clean two-stage DIA recipe.
# Stage 1: learn CON-to-evidence routing without updating the SAM mask decoder
#          or SEG fusion path.
# Stage 2: load Stage-1 model weights only, then train the full DIA mask path.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

STAGE="${STAGE:-evidence}"
case "$STAGE" in
  evidence|stage1|s1)
    export EXP_NAME="${EXP_NAME:-dia_twostage_aggressive_s1_evidence}"
    export DIA_TRAINING_STAGE="evidence"
    export NO_EVAL="${NO_EVAL:-1}"
    export EPOCHS="${EPOCHS:-12}"
    export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
    export LR="${LR:-8e-5}"
    export ATTN_LOSS_WEIGHT="${ATTN_LOSS_WEIGHT:-0.05}"
    export DIA_STAGE1_CE_LOSS_WEIGHT="${DIA_STAGE1_CE_LOSS_WEIGHT:-0.25}"
    export DIA_STAGE1_ATTN_LOSS_WEIGHT="${DIA_STAGE1_ATTN_LOSS_WEIGHT:-0.25}"
    export DIA_STAGE1_EVIDENCE_USAGE_LOSS_WEIGHT="${DIA_STAGE1_EVIDENCE_USAGE_LOSS_WEIGHT:-0.15}"
    export DIA_STAGE1_AREA_RECALL_LOSS_WEIGHT="${DIA_STAGE1_AREA_RECALL_LOSS_WEIGHT:-0.05}"
    export VISUAL_BOTTLENECK="${VISUAL_BOTTLENECK:-1}"
    export VISUAL_BOTTLENECK_BETA="${VISUAL_BOTTLENECK_BETA:-0.35}"
    export VISUAL_BOTTLENECK_MAX_DELTA_RATIO="${VISUAL_BOTTLENECK_MAX_DELTA_RATIO:-0.35}"
    export LATENT_SPARSE_MAX_DELTA_RATIO="${LATENT_SPARSE_MAX_DELTA_RATIO:-0.45}"
    export LATENT_SPARSE_DELTA_GAIN="${LATENT_SPARSE_DELTA_GAIN:-3.5}"
    export LATENT_DENSE_MAX_DELTA_RATIO="${LATENT_DENSE_MAX_DELTA_RATIO:-0.20}"
    ;;
  fusion|stage2|s2)
    export EXP_NAME="${EXP_NAME:-dia_twostage_aggressive_s2_fusion}"
    export DIA_TRAINING_STAGE="fusion"
    export NO_EVAL="${NO_EVAL:-0}"
    export EPOCHS="${EPOCHS:-60}"
    export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
    export LR="${LR:-5e-5}"
    export ATTN_LOSS_WEIGHT="${ATTN_LOSS_WEIGHT:-0.05}"
    export EVIDENCE_USAGE_LOSS_WEIGHT="${EVIDENCE_USAGE_LOSS_WEIGHT:-0.10}"
    export AREA_RECALL_LOSS_WEIGHT="${AREA_RECALL_LOSS_WEIGHT:-0.20}"
    export VISUAL_BOTTLENECK="${VISUAL_BOTTLENECK:-1}"
    export VISUAL_BOTTLENECK_BETA="${VISUAL_BOTTLENECK_BETA:-0.35}"
    export VISUAL_BOTTLENECK_MAX_DELTA_RATIO="${VISUAL_BOTTLENECK_MAX_DELTA_RATIO:-0.35}"
    export LATENT_SPARSE_MAX_DELTA_RATIO="${LATENT_SPARSE_MAX_DELTA_RATIO:-0.45}"
    export LATENT_SPARSE_DELTA_GAIN="${LATENT_SPARSE_DELTA_GAIN:-3.5}"
    export LATENT_DENSE_MAX_DELTA_RATIO="${LATENT_DENSE_MAX_DELTA_RATIO:-0.20}"
    export RESUME_MODEL_ONLY="${RESUME_MODEL_ONLY:-1}"
    export RESUME="${RESUME:-${PROJECT_ROOT}/runs/dia_twostage_aggressive_s1_evidence/ckpt_model}"
    ;;
  *)
    echo "STAGE must be evidence/stage1/s1 or fusion/stage2/s2" >&2
    exit 1
    ;;
esac

export USE_DIA="${USE_DIA:-1}"
export EXPLICIT_CON="${EXPLICIT_CON:-0}"
export DIA_FUSION_MODE="${DIA_FUSION_MODE:-latent_sparse_dense_dia}"
export DIA_NUM_EVIDENCE_TOKENS="${DIA_NUM_EVIDENCE_TOKENS:-4}"
export DIA_NUM_HEADS="${DIA_NUM_HEADS:-8}"
export DIA_ATTN_DROPOUT="${DIA_ATTN_DROPOUT:-0.0}"
export FUSION_DROPOUT="${FUSION_DROPOUT:-0.0}"
export INIT_CON_FROM_SEG="${INIT_CON_FROM_SEG:-1}"
export ALLOW_RESUME_HPARAM_MISMATCH="${ALLOW_RESUME_HPARAM_MISMATCH:-0}"

echo "Launching aggressive two-stage DIA: STAGE=${STAGE}, EXP_NAME=${EXP_NAME}"
exec bash "${PROJECT_ROOT}/train_dia_lisat.sh"
