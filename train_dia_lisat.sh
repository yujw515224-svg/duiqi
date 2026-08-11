#!/bin/bash
set -euo pipefail

# DIA-LISAt full-mix second-stage training launcher.
# Defaults match the completed LISAt baseline run except for DIA-only modules.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONDA_BIN="${CONDA_BIN:-/root/miniconda3/envs/lisa/bin}"
export PATH="$CONDA_BIN:$PATH"
DEEPSPEED_CMD=("${CONDA_BIN}/python" "${CONDA_BIN}/deepspeed")

GPU_IDS="${GPU_IDS:-0,1}"
MASTER_PORT="${MASTER_PORT:-16167}"

VERSION="${VERSION:-/root/autodl-tmp/DIA-LISAt_code/model/LISAT_PRE-7b-local-remoteclip}"
VISION_TOWER="${VISION_TOWER:-/root/autodl-tmp/DIA-LISAt_code/model/remote_clip_vit_l_14}"
VISION_PRETRAINED="${VISION_PRETRAINED:-/root/autodl-tmp/DIA-LISAt_code/sam_vit_h_4b8939.pth}"
DATASET_DIR="${DATASET_DIR:-/root/autodl-tmp/LISAt_code/dataset}"
LOG_BASE_DIR="${LOG_BASE_DIR:-/root/autodl-tmp/DIA-LISAt_code/runs}"
EXP_NAME="${EXP_NAME:-dia_evidence_feedback_v2}"

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
EVAL_SAMPLES="${EVAL_SAMPLES:-0}"
BEST_METRIC="${BEST_METRIC:-val_giou}"
MIN_BEST_SCORE_TO_SAVE="${MIN_BEST_SCORE_TO_SAVE:-1e-8}"

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
PRINT_FREQ="${PRINT_FREQ:-10}"

DIA_NUM_EVIDENCE_TOKENS="${DIA_NUM_EVIDENCE_TOKENS:-1}"
DIA_NUM_HEADS="${DIA_NUM_HEADS:-8}"
DIA_ATTN_DROPOUT="${DIA_ATTN_DROPOUT:-0.0}"
FUSION_DROPOUT="${FUSION_DROPOUT:-0.0}"
ATTN_LOSS_WEIGHT="${ATTN_LOSS_WEIGHT:-0.05}"
DIA_TRAINING_STAGE="${DIA_TRAINING_STAGE:-one_stage}"
DIA_STAGE1_CE_LOSS_WEIGHT="${DIA_STAGE1_CE_LOSS_WEIGHT:-0.25}"
DIA_STAGE1_ATTN_LOSS_WEIGHT="${DIA_STAGE1_ATTN_LOSS_WEIGHT:-0.25}"
DIA_STAGE1_EVIDENCE_USAGE_LOSS_WEIGHT="${DIA_STAGE1_EVIDENCE_USAGE_LOSS_WEIGHT:-0.10}"
DIA_STAGE1_AREA_RECALL_LOSS_WEIGHT="${DIA_STAGE1_AREA_RECALL_LOSS_WEIGHT:-0.0}"
DIA_BYPASS_FUSION="${DIA_BYPASS_FUSION:-0}"
USE_DIA="${USE_DIA:-1}"
EXPLICIT_CON="${EXPLICIT_CON:-1}"
DIA_FUSION_MODE="${DIA_FUSION_MODE:-evidence_feedback}"
TOKEN_BRIDGE_INIT_GATE="${TOKEN_BRIDGE_INIT_GATE:-0.02}"
ROLE_ADAPTER_HIDDEN_DIM="${ROLE_ADAPTER_HIDDEN_DIM:-256}"
ROLE_MAX_DELTA_RATIO="${ROLE_MAX_DELTA_RATIO:-0.05}"
DENSE_MAX_DELTA_RATIO="${DENSE_MAX_DELTA_RATIO:-0.10}"
DENSE_CONFIDENCE_POWER="${DENSE_CONFIDENCE_POWER:-0.25}"
DENSE_ATTN_CLIP="${DENSE_ATTN_CLIP:-8.0}"
FAITHFUL_FUSION_HIDDEN_DIM="${FAITHFUL_FUSION_HIDDEN_DIM:-256}"
FAITHFUL_MAX_DELTA_RATIO="${FAITHFUL_MAX_DELTA_RATIO:-0.35}"
FAITHFUL_DELTA_GAIN="${FAITHFUL_DELTA_GAIN:-2.5}"
FAITHFUL_STRICT_CONFIG="${FAITHFUL_STRICT_CONFIG:-0}"
LATENT_SPARSE_HIDDEN_DIM="${LATENT_SPARSE_HIDDEN_DIM:-256}"
LATENT_SPARSE_MAX_DELTA_RATIO="${LATENT_SPARSE_MAX_DELTA_RATIO:-0.40}"
LATENT_SPARSE_DELTA_GAIN="${LATENT_SPARSE_DELTA_GAIN:-3.0}"
LATENT_SPARSE_INIT_STD="${LATENT_SPARSE_INIT_STD:-0.001}"
LATENT_DENSE_MAX_DELTA_RATIO="${LATENT_DENSE_MAX_DELTA_RATIO:-0.15}"
LATENT_DENSE_INIT_STD="${LATENT_DENSE_INIT_STD:-0.001}"
VISUAL_BOTTLENECK="${VISUAL_BOTTLENECK:-1}"
VISUAL_BOTTLENECK_BETA="${VISUAL_BOTTLENECK_BETA:-0.30}"
VISUAL_BOTTLENECK_ATTN_CLIP="${VISUAL_BOTTLENECK_ATTN_CLIP:-8.0}"
VISUAL_BOTTLENECK_MAX_DELTA_RATIO="${VISUAL_BOTTLENECK_MAX_DELTA_RATIO:-0.20}"
VISUAL_BOTTLENECK_CONFIDENCE_POWER="${VISUAL_BOTTLENECK_CONFIDENCE_POWER:-0.25}"
VISUAL_BOTTLENECK_INIT_STD="${VISUAL_BOTTLENECK_INIT_STD:-0.001}"
EVIDENCE_USAGE_LOSS_WEIGHT="${EVIDENCE_USAGE_LOSS_WEIGHT:-0.10}"
EVIDENCE_TARGET_DELTA_RATIO="${EVIDENCE_TARGET_DELTA_RATIO:-0.12}"
AREA_RECALL_LOSS_WEIGHT="${AREA_RECALL_LOSS_WEIGHT:-0.20}"
MAP_LOSS_WEIGHT="${MAP_LOSS_WEIGHT:-0.10}"
ANCHOR_LOSS_WEIGHT="${ANCHOR_LOSS_WEIGHT:-0.10}"
ANCHOR_DECAY_STEPS="${ANCHOR_DECAY_STEPS:-8000}"
DIA_LOC_BIAS_INIT="${DIA_LOC_BIAS_INIT:--4.0}"
DIA_FUSION_MAX_STRENGTH="${DIA_FUSION_MAX_STRENGTH:-0.15}"
DIA_FUSION_WARMUP_STEPS="${DIA_FUSION_WARMUP_STEPS:-2000}"
DIA_FUSION_RAMP_STEPS="${DIA_FUSION_RAMP_STEPS:-4000}"
DIA_GATE_FLOOR="${DIA_GATE_FLOOR:-0.10}"
DIA_INIT_GATE="${DIA_INIT_GATE:-0.50}"
DIA_LR_MULTIPLIER="${DIA_LR_MULTIPLIER:-2.0}"
INIT_CON_FROM_SEG="${INIT_CON_FROM_SEG:-1}"

RUN_BACKGROUND="${RUN_BACKGROUND:-0}"
LOG_FILE="${LOG_FILE:-${LOG_BASE_DIR}/${EXP_NAME}_train.log}"
RESUME="${RESUME:-}"
RESUME_MODEL_ONLY="${RESUME_MODEL_ONLY:-0}"
AUTO_RESUME="${AUTO_RESUME:-0}"
NO_EVAL="${NO_EVAL:-0}"
ALLOW_RESUME_HPARAM_MISMATCH="${ALLOW_RESUME_HPARAM_MISMATCH:-0}"

if [ "$DIA_FUSION_MODE" = "sparse_dense" ]; then
  if [ "$USE_DIA" != "1" ] || [ "$EXPLICIT_CON" != "1" ]; then
    echo "sparse_dense requires USE_DIA=1 and EXPLICIT_CON=1" >&2
    exit 1
  fi
  if [ "$DIA_BYPASS_FUSION" = "1" ]; then
    echo "sparse_dense cannot use DIA_BYPASS_FUSION=1" >&2
    exit 1
  fi
fi

if [ "$DIA_FUSION_MODE" = "bounded_sparse_dense" ]; then
  if [ "$USE_DIA" != "1" ] || [ "$EXPLICIT_CON" != "1" ]; then
    echo "bounded_sparse_dense requires USE_DIA=1 and EXPLICIT_CON=1" >&2
    exit 1
  fi
  if [ "$DIA_BYPASS_FUSION" = "1" ]; then
    echo "bounded_sparse_dense cannot use DIA_BYPASS_FUSION=1" >&2
    exit 1
  fi
fi

if [ "$DIA_FUSION_MODE" = "faithful_evidence_fusion" ]; then
  if [ "$USE_DIA" != "1" ] || [ "$EXPLICIT_CON" != "1" ]; then
    echo "faithful_evidence_fusion requires USE_DIA=1 and EXPLICIT_CON=1" >&2
    exit 1
  fi
  if [ "$INIT_CON_FROM_SEG" != "1" ]; then
    echo "faithful_evidence_fusion requires INIT_CON_FROM_SEG=1" >&2
    exit 1
  fi
  if [ "$DIA_BYPASS_FUSION" = "1" ]; then
    echo "faithful_evidence_fusion cannot use DIA_BYPASS_FUSION=1" >&2
    exit 1
  fi
fi

if [ "$DIA_FUSION_MODE" = "evidence_feedback" ]; then
  if [ "$USE_DIA" != "1" ] || [ "$EXPLICIT_CON" != "1" ]; then
    echo "evidence_feedback requires USE_DIA=1 and EXPLICIT_CON=1" >&2
    exit 1
  fi
  if [ "$INIT_CON_FROM_SEG" != "1" ]; then
    echo "evidence_feedback requires INIT_CON_FROM_SEG=1" >&2
    exit 1
  fi
  if [ "$DIA_NUM_EVIDENCE_TOKENS" != "1" ]; then
    echo "evidence_feedback requires DIA_NUM_EVIDENCE_TOKENS=1" >&2
    exit 1
  fi
  if [ "$DIA_BYPASS_FUSION" = "1" ]; then
    echo "evidence_feedback cannot use DIA_BYPASS_FUSION=1" >&2
    exit 1
  fi
fi

if [ "$DIA_FUSION_MODE" = "latent_sparse_dense_dia" ]; then
  if [ "$USE_DIA" != "1" ]; then
    echo "latent_sparse_dense_dia requires USE_DIA=1" >&2
    exit 1
  fi
  if [ "$EXPLICIT_CON" = "1" ]; then
    echo "latent_sparse_dense_dia uses latent CON from [SEG]; set EXPLICIT_CON=0" >&2
    exit 1
  fi
  if [ "$DIA_BYPASS_FUSION" = "1" ]; then
    echo "latent_sparse_dense_dia cannot use DIA_BYPASS_FUSION=1" >&2
    exit 1
  fi
  if [ "$DIA_TRAINING_STAGE" != "one_stage" ] \
    && [ "$DIA_TRAINING_STAGE" != "evidence" ] \
    && [ "$DIA_TRAINING_STAGE" != "fusion" ]; then
    echo "DIA_TRAINING_STAGE must be one_stage, evidence, or fusion" >&2
    exit 1
  fi
fi

if [ "$DIA_TRAINING_STAGE" != "one_stage" ] \
  && [ "$DIA_FUSION_MODE" != "latent_sparse_dense_dia" ]; then
  echo "Two-stage DIA requires DIA_FUSION_MODE=latent_sparse_dense_dia" >&2
  exit 1
fi

if [ "$DIA_TRAINING_STAGE" = "fusion" ] \
  && [ -n "$RESUME" ] \
  && [ "$RESUME_MODEL_ONLY" != "1" ]; then
  echo "fusion stage should load evidence-stage checkpoints with RESUME_MODEL_ONLY=1" >&2
  exit 1
fi

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
  --eval_samples "$EVAL_SAMPLES"
  --best_metric "$BEST_METRIC"
  --min_best_score_to_save "$MIN_BEST_SCORE_TO_SAVE"
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
  --print_freq "$PRINT_FREQ"
  --save_visualizations
  --vis_samples "$VIS_SAMPLES"
)

if [ "${USE_DIA}" = "1" ]; then
  ARGS+=(
    --use_dia
    --dia_num_evidence_tokens "$DIA_NUM_EVIDENCE_TOKENS"
    --dia_num_heads "$DIA_NUM_HEADS"
    --dia_attn_dropout "$DIA_ATTN_DROPOUT"
    --fusion_dropout "$FUSION_DROPOUT"
    --attn_loss_weight "$ATTN_LOSS_WEIGHT"
    --dia_training_stage "$DIA_TRAINING_STAGE"
    --dia_stage1_ce_loss_weight "$DIA_STAGE1_CE_LOSS_WEIGHT"
    --dia_stage1_attn_loss_weight "$DIA_STAGE1_ATTN_LOSS_WEIGHT"
    --dia_stage1_evidence_usage_loss_weight "$DIA_STAGE1_EVIDENCE_USAGE_LOSS_WEIGHT"
    --dia_stage1_area_recall_loss_weight "$DIA_STAGE1_AREA_RECALL_LOSS_WEIGHT"
    --dia_fusion_mode "$DIA_FUSION_MODE"
    --token_bridge_init_gate "$TOKEN_BRIDGE_INIT_GATE"
    --role_adapter_hidden_dim "$ROLE_ADAPTER_HIDDEN_DIM"
    --role_max_delta_ratio "$ROLE_MAX_DELTA_RATIO"
    --dense_max_delta_ratio "$DENSE_MAX_DELTA_RATIO"
    --dense_confidence_power "$DENSE_CONFIDENCE_POWER"
    --dense_attn_clip "$DENSE_ATTN_CLIP"
    --faithful_fusion_hidden_dim "$FAITHFUL_FUSION_HIDDEN_DIM"
    --faithful_max_delta_ratio "$FAITHFUL_MAX_DELTA_RATIO"
    --faithful_delta_gain "$FAITHFUL_DELTA_GAIN"
    --latent_sparse_hidden_dim "$LATENT_SPARSE_HIDDEN_DIM"
    --latent_sparse_max_delta_ratio "$LATENT_SPARSE_MAX_DELTA_RATIO"
    --latent_sparse_delta_gain "$LATENT_SPARSE_DELTA_GAIN"
    --latent_sparse_init_std "$LATENT_SPARSE_INIT_STD"
    --latent_dense_max_delta_ratio "$LATENT_DENSE_MAX_DELTA_RATIO"
    --latent_dense_init_std "$LATENT_DENSE_INIT_STD"
    --visual_bottleneck_beta "$VISUAL_BOTTLENECK_BETA"
    --visual_bottleneck_attn_clip "$VISUAL_BOTTLENECK_ATTN_CLIP"
    --visual_bottleneck_max_delta_ratio "$VISUAL_BOTTLENECK_MAX_DELTA_RATIO"
    --visual_bottleneck_confidence_power "$VISUAL_BOTTLENECK_CONFIDENCE_POWER"
    --visual_bottleneck_init_std "$VISUAL_BOTTLENECK_INIT_STD"
    --evidence_usage_loss_weight "$EVIDENCE_USAGE_LOSS_WEIGHT"
    --evidence_target_delta_ratio "$EVIDENCE_TARGET_DELTA_RATIO"
    --area_recall_loss_weight "$AREA_RECALL_LOSS_WEIGHT"
    --map_loss_weight "$MAP_LOSS_WEIGHT"
    --anchor_loss_weight "$ANCHOR_LOSS_WEIGHT"
    --anchor_decay_steps "$ANCHOR_DECAY_STEPS"
    --dia_loc_bias_init "$DIA_LOC_BIAS_INIT"
    --dia_fusion_max_strength "$DIA_FUSION_MAX_STRENGTH"
    --dia_fusion_warmup_steps "$DIA_FUSION_WARMUP_STEPS"
    --dia_fusion_ramp_steps "$DIA_FUSION_RAMP_STEPS"
    --dia_gate_floor "$DIA_GATE_FLOOR"
    --dia_init_gate "$DIA_INIT_GATE"
    --dia_lr_multiplier "$DIA_LR_MULTIPLIER"
  )
  if [ "${VISUAL_BOTTLENECK}" = "1" ]; then
    ARGS+=(--visual_bottleneck_enabled)
  else
    ARGS+=(--no_visual_bottleneck)
  fi
  if [ "${DIA_BYPASS_FUSION}" = "1" ]; then
    ARGS+=(--dia_bypass_fusion)
  fi

  if [ "${EXPLICIT_CON}" = "1" ]; then
    ARGS+=(--explicit_con_in_conversation)
    if [ "${INIT_CON_FROM_SEG}" = "1" ]; then
      ARGS+=(--init_con_from_seg)
    else
      ARGS+=(--no_init_con_from_seg)
    fi
  elif [ "${INIT_CON_FROM_SEG}" = "1" ]; then
    ARGS+=(--init_con_from_seg)
  else
    ARGS+=(--no_init_con_from_seg)
  fi

  if [ "${FAITHFUL_STRICT_CONFIG}" = "1" ]; then
    ARGS+=(--faithful_strict_config)
  fi
elif [ "${EXPLICIT_CON}" = "1" ]; then
  echo "EXPLICIT_CON=1 requires USE_DIA=1" >&2
  exit 1
fi

if [ -n "$RESUME" ]; then
  ARGS+=(--resume "$RESUME")
fi

if [ "$RESUME_MODEL_ONLY" = "1" ]; then
  ARGS+=(--resume_model_only)
fi

if [ "$AUTO_RESUME" = "1" ]; then
  ARGS+=(--auto_resume)
else
  ARGS+=(--no_auto_resume)
fi

if [ "$ALLOW_RESUME_HPARAM_MISMATCH" = "1" ]; then
  ARGS+=(--allow_resume_hparam_mismatch)
fi

if [ "$NO_EVAL" = "1" ]; then
  ARGS+=(--no_eval)
fi

echo "DIA-LISAt training"
echo "  GPUs: CUDA_VISIBLE_DEVICES=${GPU_IDS}"
echo "  master_port: ${MASTER_PORT}"
echo "  exp_name: ${EXP_NAME}"
echo "  dataset_dir: ${DATASET_DIR}"
echo "  lr=${LR}, epochs=${EPOCHS}, steps_per_epoch=${STEPS_PER_EPOCH}"
echo "  batch_size=${BATCH_SIZE}, grad_accumulation_steps=${GRAD_ACCUMULATION_STEPS}"
echo "  zero_stage=${ZERO_STAGE}, zero_bucket_size=${ZERO_BUCKET_SIZE}"
echo "  use_dia=${USE_DIA}"
echo "  resume=${RESUME:-None}, resume_model_only=${RESUME_MODEL_ONLY}, auto_resume=${AUTO_RESUME}, no_eval=${NO_EVAL}"
if [ "${USE_DIA}" = "1" ]; then
  echo "  DIA: stage=${DIA_TRAINING_STAGE}, K=${DIA_NUM_EVIDENCE_TOKENS}, heads=${DIA_NUM_HEADS}, attn_dropout=${DIA_ATTN_DROPOUT}, attn_loss_weight=${ATTN_LOSS_WEIGHT}, bypass_fusion=${DIA_BYPASS_FUSION}, explicit_con=${EXPLICIT_CON}"
  if [ "$DIA_FUSION_MODE" = "evidence_feedback" ]; then
    echo "  EvidenceFeedbackV2: map_loss=${MAP_LOSS_WEIGHT}, anchor_loss=${ANCHOR_LOSS_WEIGHT}, anchor_decay=${ANCHOR_DECAY_STEPS}, loc_bias=${DIA_LOC_BIAS_INIT}, max_strength=${DIA_FUSION_MAX_STRENGTH}, warmup=${DIA_FUSION_WARMUP_STEPS}, ramp=${DIA_FUSION_RAMP_STEPS}, gate_floor=${DIA_GATE_FLOOR}, init_gate=${DIA_INIT_GATE}, dia_lr_multiplier=${DIA_LR_MULTIPLIER}"
  elif [ "$DIA_FUSION_MODE" = "latent_sparse_dense_dia" ]; then
    echo "  Stage1: ce=${DIA_STAGE1_CE_LOSS_WEIGHT}, attn=${DIA_STAGE1_ATTN_LOSS_WEIGHT}, evidence_usage=${DIA_STAGE1_EVIDENCE_USAGE_LOSS_WEIGHT}, area_recall=${DIA_STAGE1_AREA_RECALL_LOSS_WEIGHT}"
    echo "  DIA-VEB: enabled=${VISUAL_BOTTLENECK}, beta=${VISUAL_BOTTLENECK_BETA}, cap=${VISUAL_BOTTLENECK_MAX_DELTA_RATIO}, conf_power=${VISUAL_BOTTLENECK_CONFIDENCE_POWER}, latent_sparse_cap=${LATENT_SPARSE_MAX_DELTA_RATIO}, latent_dense_cap=${LATENT_DENSE_MAX_DELTA_RATIO}"
  fi
fi

mkdir -p "$LOG_BASE_DIR"

if [ "$RUN_BACKGROUND" = "1" ]; then
  echo "Running in background. Log: ${LOG_FILE}"
  nohup env CUDA_VISIBLE_DEVICES="$GPU_IDS" "${DEEPSPEED_CMD[@]}" --master_port "$MASTER_PORT" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
  echo $! > "${LOG_FILE%.log}.pid"
  echo "PID: $(cat "${LOG_FILE%.log}.pid")"
else
  exec env CUDA_VISIBLE_DEVICES="$GPU_IDS" "${DEEPSPEED_CMD[@]}" --master_port "$MASTER_PORT" "${ARGS[@]}"
fi
