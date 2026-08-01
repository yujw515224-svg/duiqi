#!/bin/bash

# Parameters. Override these with env vars if needed, e.g. BATCH_SIZE=1 EXP_NAME=xxx bash train_lisat.sh ...
VERSION="${VERSION:-/home/public/students/yujiawei/Documents/trae_projects/LISAt/model/LISAt-7b-local-remoteclip}"
VISION_TOWER="${VISION_TOWER:-/home/public/students/yujiawei/Documents/trae_projects/LISAt/model/remote_clip_vit_l_14}"
DATASET_DIR="${DATASET_DIR:-./dataset}"
EXP_NAME="${EXP_NAME:-lisat_refsegrs}"

# RefSegRS baseline configuration
DATASET_REFERSEG="${DATASET_REFERSEG:-refsegrs}"
SAMPLE_RATES_REFERSEG="${SAMPLE_RATES_REFERSEG:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
NUM_CLASSES_PER_SAMPLE="${NUM_CLASSES_PER_SAMPLE:-1}"
EVAL_SPLIT="${4:-val}"  # val or test; use val during training, test only for final reporting
EVAL_SAMPLES="${EVAL_SAMPLES:-0}"  # 0 means full eval split
MASK_THRESHOLD="${MASK_THRESHOLD:-0.0}"
BEST_METRIC="${BEST_METRIC:-val_giou}"  # val_giou, val_ciou, val_miou, val_acc, val_f1, or combined
GPU_MEM_WARN_MB="${GPU_MEM_WARN_MB:-1000}"

# User-defined parameters
TRAINING_TYPE="$1"  # ReferSeg, ReferSegEval, or ReasonSeg
GPU_REQUEST="$2"    # GPU count: 1/2, explicit IDs: 0,1, or old DeepSpeed include: localhost:1
MASTER_PORT="$3"    # e.g., 15990

function usage() {
    echo "Usage: $0 <Training Type> <GPU Count|GPU IDs|DeepSpeed Include> <Master Port> [Eval Split]"
    echo "Examples:"
    echo "  Auto-pick 1 least-used GPU:       $0 ReferSeg 1 15990 val"
    echo "  Auto-pick 2 least-used GPUs:      $0 ReferSeg 2 15990 val"
    echo "  Use exact GPU IDs 0 and 1:        $0 ReferSeg 0,1 15990 val"
    echo "  Use old explicit include syntax:  $0 ReferSeg localhost:1 15990 val"
    echo "  Evaluate RefSegRS test only:      $0 ReferSegEval 1 15991 test"
}

function resolve_gpu_settings() {
    local request="$1"
    if [[ "$request" == localhost:* ]]; then
        echo "$request"
        return 0
    fi

    if [[ "$request" =~ ^[0-9]+$ ]]; then
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "localhost:0"
            return 0
        fi
        local total
        total=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
        if [ "$request" -lt 1 ] || [ "$request" -gt "$total" ]; then
            echo "Requested $request GPU(s), but this machine has $total GPU(s)." >&2
            return 1
        fi
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | sort -t, -k2,2n \
            | head -n "$request" \
            | awk -F, '{gsub(/ /, "", $1); ids = ids (NR == 1 ? "" : ",") $1} END {print "localhost:" ids}'
        return 0
    fi

    if [[ "$request" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "localhost:$request"
        return 0
    fi

    echo "$request"
}

function count_selected_gpus() {
    local setting="${1#localhost:}"
    if [ -z "$setting" ]; then
        echo 0
        return 0
    fi
    awk -F, '{print NF}' <<< "$setting"
}

function warn_selected_gpu_memory() {
    local setting="${1#localhost:}"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    IFS=',' read -ra gpu_ids <<< "$setting"
    for gpu_id in "${gpu_ids[@]}"; do
        local used
        used=$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ -n "$used" ] && [ "$used" -gt "$GPU_MEM_WARN_MB" ]; then
            echo "WARNING: GPU ${gpu_id} already uses ${used} MiB. LISAt is memory-heavy; this GPU may OOM."
        fi
    done
}

if [ -z "$TRAINING_TYPE" ] || [ -z "$GPU_REQUEST" ] || [ -z "$MASTER_PORT" ]; then
    usage
    exit 1
fi

if [ "$EVAL_SPLIT" != "val" ] && [ "$EVAL_SPLIT" != "test" ]; then
    echo "Invalid Eval Split: $EVAL_SPLIT. Use 'val' or 'test'."
    exit 1
fi

if [[ ! "$BEST_METRIC" =~ ^(val_giou|val_ciou|val_miou|val_acc|val_f1|combined)$ ]]; then
    echo "Invalid BEST_METRIC: $BEST_METRIC. Use val_giou, val_ciou, val_miou, val_acc, val_f1, or combined."
    exit 1
fi

GPU_SETTINGS=$(resolve_gpu_settings "$GPU_REQUEST") || exit 1
GPU_COUNT=$(count_selected_gpus "$GPU_SETTINGS")
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUMULATION_STEPS * GPU_COUNT))

echo "GPU request '${GPU_REQUEST}' resolved to '${GPU_SETTINGS}' (${GPU_COUNT} GPU(s))."
echo "Per-GPU batch=${BATCH_SIZE}, grad_accumulation=${GRAD_ACCUMULATION_STEPS}, effective batch=${EFFECTIVE_BATCH}."
echo "Best checkpoint metric=${BEST_METRIC}."
echo "Mask threshold=${MASK_THRESHOLD}."
warn_selected_gpu_memory "$GPU_SETTINGS"

# Vision backbone checkpoint
VISION_PRETRAINED="${VISION_PRETRAINED:-./sam_vit_h_4b8939.pth}"
URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

if [ ! -f "$VISION_PRETRAINED" ]; then
    echo "SAM model not found, downloading..."
    wget -O "$VISION_PRETRAINED" "$URL"
else
    echo "SAM model already exists."
fi

# ReasonSeg Configuration
DATASET_REASONSEG="${DATASET_REASONSEG:-sem_seg||refer_seg||correct_refer_seg||vqa||neg_refer_seg||reason_seg||geo_reason_seg}"
SAMPLE_RATES_REASONSEG="${SAMPLE_RATES_REASONSEG:-15,15,2,30,1,1,36}"

if [ "$TRAINING_TYPE" == "ReferSeg" ]; then
    echo "Launching RefSegRS training with ${EVAL_SPLIT} evaluation..."
    deepspeed --include "$GPU_SETTINGS" --master_port="$MASTER_PORT" train_lisat.py \
      --version="$VERSION" \
      --vision-tower="$VISION_TOWER" \
      --dataset_dir="$DATASET_DIR" \
      --vision_pretrained="$VISION_PRETRAINED" \
      --exp_name="$EXP_NAME" \
      --dataset="$DATASET_REFERSEG" \
      --sample_rates="$SAMPLE_RATES_REFERSEG" \
      --batch_size=$BATCH_SIZE \
      --grad_accumulation_steps $GRAD_ACCUMULATION_STEPS \
      --num_classes_per_sample=$NUM_CLASSES_PER_SAMPLE \
      --eval_dataset="refsegrs" \
      --eval_split="$EVAL_SPLIT" \
      --eval_samples="$EVAL_SAMPLES" \
      --mask_threshold="$MASK_THRESHOLD" \
      --best_metric="$BEST_METRIC"

elif [ "$TRAINING_TYPE" == "ReferSegEval" ]; then
    echo "Launching RefSegRS ${EVAL_SPLIT} evaluation only..."
    RESUME_DIR="${RESUME_DIR:-./runs/${EXP_NAME}/best_ckpt_model}"
    if [ ! -e "$RESUME_DIR" ] && [ -e "./runs/${EXP_NAME}/ckpt_model" ]; then
      RESUME_DIR="./runs/${EXP_NAME}/ckpt_model"
    fi
    echo "Resume checkpoint=${RESUME_DIR}"
    deepspeed --include "$GPU_SETTINGS" --master_port="$MASTER_PORT" train_lisat.py \
      --version="$VERSION" \
      --vision-tower="$VISION_TOWER" \
      --dataset_dir="$DATASET_DIR" \
      --vision_pretrained="$VISION_PRETRAINED" \
      --exp_name="$EXP_NAME" \
      --dataset="$DATASET_REFERSEG" \
      --sample_rates="$SAMPLE_RATES_REFERSEG" \
      --batch_size=$BATCH_SIZE \
      --grad_accumulation_steps $GRAD_ACCUMULATION_STEPS \
      --num_classes_per_sample=$NUM_CLASSES_PER_SAMPLE \
      --eval_dataset="refsegrs" \
      --eval_split="$EVAL_SPLIT" \
      --eval_samples="$EVAL_SAMPLES" \
      --mask_threshold="$MASK_THRESHOLD" \
      --best_metric="$BEST_METRIC" \
      --resume="$RESUME_DIR" \
      --eval_only

elif [ "$TRAINING_TYPE" == "ReasonSeg" ]; then
    echo "Launching ReasonSeg training..."
    deepspeed --include "$GPU_SETTINGS" --master_port="$MASTER_PORT" train_lisat.py \
      --version="$VERSION" \
      --vision-tower="$VISION_TOWER" \
      --dataset_dir="$DATASET_DIR" \
      --vision_pretrained="$VISION_PRETRAINED" \
      --exp_name="$EXP_NAME" \
      --dataset="$DATASET_REASONSEG" \
      --sample_rates="$SAMPLE_RATES_REASONSEG" \
      --reason_seg_data="ReasonSeg|train" \
      --geo_reason_seg_data="GeoReasonSeg|train" \
      --batch_size=$BATCH_SIZE \
      --grad_accumulation_steps $GRAD_ACCUMULATION_STEPS \
      --num_classes_per_sample=$NUM_CLASSES_PER_SAMPLE \
      --no_eval

else
    echo "Invalid training type. Use ReferSeg, ReferSegEval, or ReasonSeg."
    exit 1
fi
