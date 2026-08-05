#!/bin/bash
set -euo pipefail

# DIA-LISAt full-mix second-stage training launcher.
# Defaults match the completed LISAt baseline run except for DIA-only modules.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONDA_BIN="${CONDA_BIN:-/root/miniconda3/envs/lisa/bin}"
export PATH="$CONDA_BIN:$PATH"

GPU_IDS="${GPU_IDS:-0,1}"
MASTER_PORT="${MASTER_PORT:-16167}"

VERSION="${VERSION:-/root/autodl-tmp/DIA-LISAt_code/model/LISAT_PRE-7b-local-remoteclip}"
VISION_TOWER="${VISION_TOWER:-/root/autodl-tmp/DIA-LISAt_code/model/remote_clip_vit_l_14}"
VISION_PRETRAINED="${VISION_PRETRAINED:-/root/autodl-tmp/DIA-LISAt_code/sam_vit_h_4b8939.pth}"
DATASET_DIR="${DATASET_DIR:-/root/autodl-tmp/LISAt_code/dataset}"
LOG_BASE_DIR="${LOG_BASE_DIR:-/root/autodl-tmp/DIA-LISAt_code/runs}"
EXP_NAME="${EXP_NAME:-dia_lisat_lr5e_reinit}"

DATASET="${DATASET:-sem_seg||refer_seg||correct_refer_seg||vqa||neg_refer_seg||reason_seg||geo_reason_seg}"
SAMPLE_RATES="${SAMPLE_RATES:-15,15,2,30,1,1,36}"
SEM_SEG_DATA="${SEM_SEG_DATA:-ade20k||cocostuff||pascal_part||paco_lvis}"
REFER_SEG_DATA="${REFER_SEG_DATA:-refclef||refcoco||refcoco+||refcocog}"
NEG_REFER_SEG_DATA="${NEG_REFER_SEG_DATA:-R-refcocog||R-refcoco||R-refcoco+}"
CORRECT_REFER_SEG_DATA="${CORRECT_REFER_SEG_DATA:-fprefcocog||fprefcoco||fprefcoco+}"
VQA_DATA="${VQA_DATA:-llava_instruct_150k}"
REASON_SEG_DATA="${REASON_SEG_DATA:-ReasonSeg|train}"
GEO_REASON_SEG_DATA="${GEO_REASON_SEG_DATA:-GeoReasonSeg|train}"

EVAL_DATASET="${EVAL_DATASET:-geo_reason_seg}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
BEST_METRIC="${BEST_METRIC:-val_giou}"

BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-4}"
NUM_CLASSES_PER_SAMPLE="${NUM_CLASSES_PER_SAMPLE:-3}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}"
EPOCHS="${EPOCHS:-60}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
LR="${LR:-5e-5}"
ZERO_STAGE="${ZERO_STAGE:-2}"
ZERO_BUCKET_SIZE="${ZERO_BUCKET_SIZE:-5e7}"
WORKERS="${WORKERS:-4}"
VIS_SAMPLES="${VIS_SAMPLES:-16}"

DIA_NUM_EVIDENCE_TOKENS="${DIA_NUM_EVIDENCE_TOKENS:-1}"
DIA_NUM_HEADS="${DIA_NUM_HEADS:-8}"
DIA_ATTN_DROPOUT="${DIA_ATTN_DROPOUT:-0.0}"
FUSION_DROPOUT="${FUSION_DROPOUT:-0.0}"
ATTN_LOSS_WEIGHT="${ATTN_LOSS_WEIGHT:-0.02}"

RUN_BACKGROUND="${RUN_BACKGROUND:-0}"
LOG_FILE="${LOG_FILE:-${LOG_BASE_DIR}/${EXP_NAME}_train.log}"
RESUME="${RESUME:-}"
AUTO_RESUME="${AUTO_RESUME:-0}"

ARGS=(
  train_lisat.py
  --version "$VERSION"
  --vision-tower "$VISION_TOWER"
  --vision_pretrained "$VISION_PRETRAINED"
  --dataset_dir "$DATASET_DIR"
  --log_base_dir "$LOG_BASE_DIR"
  --exp_name "$EXP_NAME"
  --dataset "$DATASET"
  --sample_rates "$SAMPLE_RATES"
  --sem_seg_data "$SEM_SEG_DATA"
  --refer_seg_data "$REFER_SEG_DATA"
  --neg_refer_seg_data "$NEG_REFER_SEG_DATA"
  --correct_refer_seg_data "$CORRECT_REFER_SEG_DATA"
  --vqa_data "$VQA_DATA"
  --reason_seg_data "$REASON_SEG_DATA"
  --geo_reason_seg_data "$GEO_REASON_SEG_DATA"
  --eval_dataset "$EVAL_DATASET"
  --eval_split "$EVAL_SPLIT"
  --best_metric "$BEST_METRIC"
  --batch_size "$BATCH_SIZE"
  --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS"
  --num_classes_per_sample "$NUM_CLASSES_PER_SAMPLE"
  --model_max_length "$MODEL_MAX_LENGTH"
  --epochs "$EPOCHS"
  --steps_per_epoch "$STEPS_PER_EPOCH"
  --lr "$LR"
  --zero_stage "$ZERO_STAGE"
  --zero_bucket_size "$ZERO_BUCKET_SIZE"
  --workers "$WORKERS"
  --save_visualizations
  --vis_samples "$VIS_SAMPLES"
  --dia_num_evidence_tokens "$DIA_NUM_EVIDENCE_TOKENS"
  --dia_num_heads "$DIA_NUM_HEADS"
  --dia_attn_dropout "$DIA_ATTN_DROPOUT"
  --fusion_dropout "$FUSION_DROPOUT"
  --attn_loss_weight "$ATTN_LOSS_WEIGHT"
  --init_con_from_seg
)

if [ -n "$RESUME" ]; then
  ARGS+=(--resume "$RESUME")
fi

if [ "$AUTO_RESUME" = "1" ]; then
  ARGS+=(--auto_resume)
else
  ARGS+=(--no_auto_resume)
fi

echo "DIA-LISAt training"
echo "  GPUs: CUDA_VISIBLE_DEVICES=${GPU_IDS}"
echo "  master_port: ${MASTER_PORT}"
echo "  exp_name: ${EXP_NAME}"
echo "  dataset_dir: ${DATASET_DIR}"
echo "  lr=${LR}, epochs=${EPOCHS}, steps_per_epoch=${STEPS_PER_EPOCH}"
echo "  batch_size=${BATCH_SIZE}, grad_accumulation_steps=${GRAD_ACCUMULATION_STEPS}"
echo "  zero_stage=${ZERO_STAGE}, zero_bucket_size=${ZERO_BUCKET_SIZE}"
echo "  DIA: K=${DIA_NUM_EVIDENCE_TOKENS}, heads=${DIA_NUM_HEADS}, attn_dropout=${DIA_ATTN_DROPOUT}, attn_loss_weight=${ATTN_LOSS_WEIGHT}"

mkdir -p "$LOG_BASE_DIR"

if [ "$RUN_BACKGROUND" = "1" ]; then
  echo "Running in background. Log: ${LOG_FILE}"
  nohup env CUDA_VISIBLE_DEVICES="$GPU_IDS" deepspeed --master_port "$MASTER_PORT" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
  echo $! > "${LOG_FILE%.log}.pid"
  echo "PID: $(cat "${LOG_FILE%.log}.pid")"
else
  exec env CUDA_VISIBLE_DEVICES="$GPU_IDS" deepspeed --master_port "$MASTER_PORT" "${ARGS[@]}"
fi
