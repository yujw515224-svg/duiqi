# Launch flash attention (optional)

# from model.llava.train.llama_flash_attn_monkey_patch import (
# replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()


import argparse
import os
import shutil
import sys
from functools import partial
import deepspeed
import torch
import tqdm
try:
    import wandb
except ImportError:
    class _NoOpWandb:
        def init(self, *args, **kwargs):
            print("[WARN] wandb is not installed; experiment logging is disabled.")
            return self
        def log(self, *args, **kwargs):
            return None
    wandb = _NoOpWandb()
import csv
import json
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw

# Model & Data
from model.LISAT import init_LISAT_model
from model.llava import conversation as conversation_lib
from dataloaders.trainval_dataset import (
    HybridDataset, ReasonSegDataset,
    collate_fn_train, collate_fn_val
)
from dataloaders.utils import replace_image_tokens, tokenize_and_pad
from dataloaders.base_dataset import ImageProcessor
from model.llava.constants import DEFAULT_IMAGE_TOKEN

# Utils
from utils import (
    AverageMeter, ProgressMeter, Summary,
    prepare_input, intersectionAndUnionGPU,
)

# BLEU scoring
try:
    from pycocoevalcap.bleu.bleu import Bleu
except ImportError:
    Bleu = None



METRIC_FIELDS = [
    "timestamp",
    "phase",
    "dia_training_stage",
    "epoch",
    "step",
    "split",
    "loss",
    "ce_loss",
    "mask_bce_loss",
    "mask_dice_loss",
    "mask_loss",
    "attn_alignment_loss",
    "attn_loss",
    "evidence_map_loss",
    "anchor_loss",
    "anchor_loss_weight",
    "num_positive_masks",
    "num_valid_attn_masks",
    "num_valid_evidence_masks",
    "res_scale",
    "gate_mean",
    "attention_entropy",
    "map_prob_mean",
    "map_prob_max",
    "fusion_strength",
    "explicit_con_count",
    "explicit_seg_count",
    "explicit_pair_rate",
    "explicit_hidden_cosine",
    "token_gate_mean",
    "token_delta_ratio",
    "dense_delta_ratio",
    "role_preclip_ratio",
    "role_delta_ratio",
    "role_bound_scale",
    "role_bound_hit_rate",
    "bounded_dense_preclip_ratio",
    "bounded_dense_delta_ratio",
    "bounded_dense_bound_scale",
    "bounded_dense_bound_hit_rate",
    "dense_confidence",
    "dense_normalized_entropy",
    "dense_relative_attention_abs_mean",
    "faithful_raw_delta_ratio",
    "evidence_delta_ratio",
    "faithful_smooth_scale",
    "latent_sparse_raw_delta_ratio",
    "latent_sparse_delta_ratio",
    "latent_sparse_bound_scale",
    "latent_sparse_bound_hit_rate",
    "latent_dense_preclip_ratio",
    "latent_dense_delta_ratio",
    "latent_dense_bound_scale",
    "latent_dense_bound_hit_rate",
    "latent_dense_confidence",
    "visual_bottleneck_gate_mean",
    "visual_bottleneck_confidence",
    "visual_bottleneck_image_delta_ratio",
    "visual_bottleneck_residual_ratio",
    "visual_bottleneck_total_delta_ratio",
    "visual_bottleneck_bound_scale",
    "visual_bottleneck_bound_hit_rate",
    "evidence_usage_loss",
    "area_recall_loss",
    "latent_con_count",
    "attention_normalized_entropy",
    "explicit_paired_count",
    "explicit_orphan_con_count",
    "explicit_orphan_seg_count",
    "explicit_invalid_row_count",
    "sam_prompt_encoder_calls",
    "sam_mask_decoder_calls",
    "valid_attention_entropy",
    "lr",
    "val_giou",
    "val_ciou",
    "val_miou",
    "val_acc",
    "val_macc",
    "val_fg_acc",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_dice",
    "val_iou_bg",
    "val_iou_fg",
    "val_pr_50",
    "val_pr_60",
    "val_pr_70",
    "val_pr_80",
    "val_pr_90",
    "val_gt_fg_frac",
    "val_pred_fg_frac",
    "val_pred_gt_fg_ratio",
    "val_empty_gt_rate",
    "val_empty_pred_rate",
    "val_logit_mean",
    "val_mask_threshold",
    "nwpu_bleu4",
    "sydney_bleu4",
    "ucm_bleu4",
    "best_metric",
    "best_score",
    "best_epoch",
    "is_best",
    "checkpoint_dir",
]


def _metric_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {k: _metric_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metric_value(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _stringify_config_value(value):
    value = _metric_value(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def save_run_args(args):
    if getattr(args, "local_rank", 0) != 0:
        return
    os.makedirs(args.log_dir, exist_ok=True)
    args_dict = {k: _metric_value(v) for k, v in vars(args).items()}

    args_path = os.path.join(args.log_dir, "args.json")
    if os.path.exists(args_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args_path = os.path.join(args.log_dir, f"args_{timestamp}.json")
    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(
            args_dict,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )

    # Keep the DIA run configuration in the same CSV format as the baseline.
    config_path = os.path.join(args.log_dir, "run_config.csv")
    if os.path.exists(config_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = os.path.join(args.log_dir, f"run_config_{timestamp}.csv")
    with open(config_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "value"])
        writer.writeheader()
        for name in sorted(args_dict):
            writer.writerow({"name": name, "value": _stringify_config_value(args_dict[name])})


def _check_resume_mode_compatibility(args):
    """Best-effort guard against mixing baseline and DIA checkpoints."""
    if not args.resume:
        return

    resume_dir = args.resume
    candidates = [
        os.path.join(resume_dir, "args.json"),
        os.path.join(os.path.dirname(resume_dir), "args.json"),
    ]
    for args_path in candidates:
        if not os.path.exists(args_path):
            continue
        with open(args_path, "r", encoding="utf-8") as handle:
            saved_args = json.load(handle)
        if "use_dia" not in saved_args:
            return
        saved_use_dia = bool(saved_args["use_dia"])
        if saved_use_dia != bool(args.use_dia):
            raise RuntimeError(
                "Resume mode mismatch: checkpoint was created with "
                f"use_dia={saved_use_dia}, but current run has "
                f"use_dia={args.use_dia}."
            )
        structural_comparisons = {
            "dia_fusion_mode": str,
            "explicit_con_in_conversation": bool,
            "dia_num_evidence_tokens": int,
            "dia_num_heads": int,
            "faithful_fusion_hidden_dim": int,
            "latent_sparse_hidden_dim": int,
            "visual_bottleneck_enabled": bool,
            "anchor_decay_steps": int,
        }
        tunable_comparisons = {
            "faithful_max_delta_ratio": float,
            "faithful_delta_gain": float,
            "attn_loss_weight": float,
            "latent_sparse_max_delta_ratio": float,
            "latent_sparse_delta_gain": float,
            "latent_sparse_init_std": float,
            "latent_dense_max_delta_ratio": float,
            "latent_dense_init_std": float,
            "evidence_usage_loss_weight": float,
            "evidence_target_delta_ratio": float,
            "area_recall_loss_weight": float,
            "visual_bottleneck_beta": float,
            "visual_bottleneck_attn_clip": float,
            "visual_bottleneck_max_delta_ratio": float,
            "visual_bottleneck_confidence_power": float,
            "visual_bottleneck_init_std": float,
            "map_loss_weight": float,
            "anchor_loss_weight": float,
            "dia_loc_bias_init": float,
            "dia_fusion_max_strength": float,
            "dia_fusion_warmup_steps": float,
            "dia_fusion_ramp_steps": float,
            "dia_gate_floor": float,
            "dia_init_gate": float,
        }
        comparisons = {
            **structural_comparisons,
            **tunable_comparisons,
        }
        for key, caster in comparisons.items():
            if key not in saved_args:
                continue
            saved_value = caster(saved_args[key])
            current_value = caster(getattr(args, key))
            if caster is float:
                mismatch = abs(saved_value - current_value) > 1e-12
            else:
                mismatch = saved_value != current_value
            if mismatch:
                if (
                    key in tunable_comparisons
                    and getattr(args, "allow_resume_hparam_mismatch", False)
                ):
                    if getattr(args, "local_rank", 0) == 0:
                        print(
                            "[Resume][WARN] Continuing from checkpoint with "
                            f"changed tunable DIA hparam {key}: "
                            f"saved={saved_value}, current={current_value}."
                        )
                    continue
                raise RuntimeError(
                    "Resume mode mismatch: checkpoint was created with "
                    f"{key}={saved_value}, but current run has "
                    f"{key}={current_value}."
                )
        return


def append_metrics(args, metrics):
    if getattr(args, "local_rank", 0) != 0:
        return
    os.makedirs(args.log_dir, exist_ok=True)
    record = {k: _metric_value(v) for k, v in metrics.items()}
    record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    jsonl_path = os.path.join(args.log_dir, "metrics.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")

    csv_path = os.path.join(args.log_dir, "metrics.csv")
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in METRIC_FIELDS})


def build_optimizer_param_groups(model, base_lr, dia_lr_multiplier):
    dia_keywords = (
        "con_hidden_fcs",
        "context_adapter",
        "evidence_fusion",
    )
    base_params = []
    dia_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in dia_keywords):
            dia_params.append(param)
        else:
            base_params.append(param)

    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})
    if dia_params:
        groups.append({"params": dia_params, "lr": base_lr * dia_lr_multiplier})
    return groups


def _distributed_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _remove_path(path):
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def _safe_div(numerator, denominator):
    if abs(float(denominator)) < 1e-10:
        return 0.0
    return float(numerator) / float(denominator)


def _mean_valid(values, valid_mask):
    values = np.asarray(values, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not valid_mask.any():
        return 0.0
    return float(values[valid_mask].mean())


def _mask_to_numpy(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[0]
    return (mask > 0).astype(np.uint8)


def _resize_mask(mask, size):
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img = mask_img.resize(size, Image.Resampling.NEAREST)
    return (np.asarray(mask_img) > 0).astype(np.uint8)


def _blend_mask(image, mask, color, alpha=0.45):
    arr = np.asarray(image.convert("RGB")).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    hit = mask.astype(bool)
    arr[hit] = (arr[hit].astype(np.float32) * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)
    return Image.fromarray(arr)


def _compare_masks(image, gt_mask, pred_mask):
    arr = np.asarray(image.convert("RGB")).copy()
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)
    tp = gt & pred
    fp = ~gt & pred
    fn = gt & ~pred
    arr[tp] = (255, 220, 0)
    arr[fp] = (255, 64, 64)
    arr[fn] = (40, 210, 120)
    return Image.fromarray(arr)


def _add_panel_label(panel, label):
    label_h = 24
    canvas = Image.new("RGB", (panel.width, panel.height + label_h), "white")
    canvas.paste(panel, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), label, fill=(0, 0, 0))
    return canvas


def _resize_panel(panel, max_side=384):
    panel = panel.copy()
    panel.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return panel


def _save_mask_visualization(image_path, gt_mask, pred_mask, save_path):
    image = Image.open(image_path).convert("RGB")
    gt_mask = _resize_mask(_mask_to_numpy(gt_mask), image.size)
    pred_mask = _resize_mask(_mask_to_numpy(pred_mask), image.size)

    panels = [
        _add_panel_label(_resize_panel(image), "Image"),
        _add_panel_label(_resize_panel(_blend_mask(image, gt_mask, (40, 210, 120))), "GT"),
        _add_panel_label(_resize_panel(_blend_mask(image, pred_mask, (255, 64, 64))), "Pred"),
        _add_panel_label(_resize_panel(_compare_masks(image, gt_mask, pred_mask)), "TP yellow / FP red / FN green"),
    ]
    height = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)


def _visualization_dir(args, global_iters):
    if getattr(args, "vis_dir", ""):
        base_dir = args.vis_dir
    else:
        base_dir = os.path.join(args.log_dir, "visualizations")
    split = getattr(args, "eval_split", "val")
    step_name = f"step_{int(global_iters):06d}"
    return os.path.join(base_dir, split, step_name)


def _save_seg_visualizations(input_dict, gt_masks, pred_binary_masks, args, global_iters, saved_count):
    if not getattr(args, "save_visualizations", False):
        return saved_count
    if getattr(args, "local_rank", 0) != 0 or saved_count >= getattr(args, "vis_samples", 0):
        return saved_count

    image_paths = input_dict.get("image_paths", [])
    if not image_paths:
        return saved_count

    image_path = image_paths[0]
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        return saved_count

    out_dir = _visualization_dir(args, global_iters)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    num_masks = min(len(gt_masks), len(pred_binary_masks))
    for mask_idx in range(num_masks):
        if saved_count >= args.vis_samples:
            break
        save_name = f"{saved_count:04d}_{basename}_q{mask_idx}.png"
        save_path = os.path.join(out_dir, save_name)
        _save_mask_visualization(image_path, gt_masks[mask_idx], pred_binary_masks[mask_idx], save_path)
        saved_count += 1
    return saved_count


def _write_best_metadata(args, metrics):
    if getattr(args, "local_rank", 0) != 0:
        return
    metadata_path = os.path.join(args.log_dir, "best_metrics.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {k: _metric_value(v) for k, v in metrics.items()},
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )


def save_best_checkpoint(model_engine, args, epoch, global_iters, metrics):
    save_dir = os.path.join(args.log_dir, "best_ckpt_model")
    compat_dir = os.path.join(args.log_dir, "ckpt_model")
    if args.local_rank == 0:
        _remove_path(save_dir)
    _distributed_barrier()

    client_state = {
        "epoch": epoch,
        "global_iters": global_iters,
        "best_metric": metrics.get("best_metric"),
        "best_score": metrics.get("best_score"),
    }
    model_engine.save_checkpoint(save_dir, client_state=client_state)
    _distributed_barrier()

    if args.local_rank == 0:
        _remove_path(compat_dir)
        os.symlink("best_ckpt_model", compat_dir, target_is_directory=True)
        best_record = dict(metrics)
        best_record.update(
            {
                "epoch": epoch,
                "step": global_iters,
                "checkpoint_dir": save_dir,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        _write_best_metadata(args, best_record)
        print(
            f"[Best] epoch={epoch}, step={global_iters}, "
            f"{metrics.get('best_metric')}={metrics.get('best_score'):.4f}, "
            f"saved to {save_dir}"
        )
    _distributed_barrier()


def parse_args(args):
    """Define and parse training arguments."""
    parser = argparse.ArgumentParser(description="Train DIA-LISAt Model")

    # Model paths
    parser.add_argument("--version", default="/root/autodl-tmp/DIA-LISAt_code/model/LISAT_PRE-7b-local-remoteclip")
    parser.add_argument("--vision-tower", default="/root/autodl-tmp/DIA-LISAt_code/model/remote_clip_vit_l_14")
    # Precision settings
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")

    # Image and input settings
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--model_max_length", type=int, default=1024)

    # Dataset and training configuration
    parser.add_argument(
        "--dataset",
        default="sem_seg||refer_seg||correct_refer_seg||vqa||neg_refer_seg||reason_seg||geo_reason_seg",
    )
    parser.add_argument("--sample_rates", default="15,15,2,30,1,1,36")
    parser.add_argument("--sem_seg_data", default="ade20k||cocostuff||pascal_part||paco_lvis")
    parser.add_argument("--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog")
    parser.add_argument("--neg_refer_seg_data", default="R-refcocog||R-refcoco||R-refcoco+")
    parser.add_argument("--correct_refer_seg_data", default="fprefcocog||fprefcoco||fprefcoco+")
    parser.add_argument("--vqa_data", default="llava_instruct_150k")
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train")
    parser.add_argument("--geo_reason_seg_data", default="GeoReasonSeg|train")
    parser.add_argument(
        "--dia_align_aug_source_dataset",
        default="sem_seg||refer_seg||reason_seg||geo_reason_seg",
        help="Source segmentation datasets wrapped by dia_align_aug.",
    )
    parser.add_argument(
        "--dia_align_aug_source_rates",
        default="15,15,1,36",
        help="Sampling rates for --dia_align_aug_source_dataset.",
    )
    parser.add_argument(
        "--dia_align_aug_positive_prob",
        type=float,
        default=0.5,
        help="Probability of target-preserved density-reduced DIA alignment samples.",
    )
    parser.add_argument(
        "--dia_align_aug_background_scale",
        type=float,
        default=0.2,
        help="Scale applied to non-target pixels in positive density-reduced samples.",
    )
    parser.add_argument(
        "--dia_align_aug_mask_dilation",
        type=int,
        default=2,
        help="Mask dilation in preprocessed image space for DIA edited views.",
    )
    parser.add_argument("--eval_dataset", choices=["auto", "refsegrs", "geo_reason_seg"], default="geo_reason_seg")
    parser.add_argument("--eval_split", choices=["val", "test"], default="val")
    parser.add_argument("--eval_samples", type=int, default=0, help="0 means evaluate the full split")
    parser.add_argument("--mask_threshold", type=float, default=0.0, help="Logit threshold used to binarize predicted masks during evaluation.")
    parser.add_argument(
        "--best_metric",
        choices=["val_giou", "val_ciou", "val_miou", "val_acc", "val_f1", "combined"],
        default="val_giou",
        help="Metric used to choose the best epoch checkpoint.",
    )
    parser.add_argument(
        "--min_best_score_to_save",
        type=float,
        default=1e-8,
        help="Skip checkpoint saving until the selected best metric is positive.",
    )
    parser.add_argument("--save_visualizations", action="store_true", default=True, help="Save GT/prediction mask visualizations during validation.")
    parser.add_argument("--vis_samples", type=int, default=16, help="Maximum visualizations saved per validation call.")
    parser.add_argument("--vis_dir", default="", help="Optional visualization output directory. Defaults to log_dir/visualizations.")

    # Directories
    parser.add_argument("--dataset_dir", default="/root/autodl-tmp/LISAt_code/dataset")
    parser.add_argument("--log_base_dir", default="/root/autodl-tmp/DIA-LISAt_code/runs")
    parser.add_argument("--exp_name", default="dia_lisat_lr5e")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=4)
    parser.add_argument("--zero_stage", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--zero_bucket_size", type=float, default=5e7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--dice_loss_weight", type=float, default=0.5)
    parser.add_argument("--bce_loss_weight", type=float, default=2.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--num_classes_per_sample", type=int, default=3)

    # Training control
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--resume_model_only",
        action="store_true",
        default=False,
        help=(
            "Load only model weights from --resume. Use this when switching "
            "between DIA stages because the trainable parameter set changes."
        ),
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--auto_resume",
        dest="auto_resume",
        action="store_true",
    )
    resume_group.add_argument(
        "--no_auto_resume",
        dest="auto_resume",
        action="store_false",
    )
    parser.set_defaults(auto_resume=False)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--conv_type", choices=["llava_v1", "llava_llama_2"], default="llava_v1")
    parser.add_argument("--vision_pretrained", default="/root/autodl-tmp/DIA-LISAt_code/sam_vit_h_4b8939.pth")
    parser.add_argument("--out_dim", type=int, default=256)

    # Eval VQA captioning files
    parser.add_argument("--vqa_eval_file_nwpu", default="./dataset/vqa_caption/NWPU-Captions.jsonl")
    parser.add_argument("--vqa_eval_file_sydney", default="./dataset/vqa_caption/Sydney-Captions.jsonl")
    parser.add_argument("--vqa_eval_file_ucm", default="./dataset/vqa_caption/UCM-Captions.jsonl")

    parser.add_argument(
        "--use_dia",
        action="store_true",
        default=False,
        help="Enable DIA-LISAt modules. Default False keeps the original LISAt path.",
    )
    parser.add_argument(
        "--explicit_con_in_conversation",
        action="store_true",
        default=False,
        help="Train explicit DIA answers as adjacent [CON][SEG] token pairs.",
    )
    parser.add_argument("--attn_loss_weight", type=float, default=0.02)
    parser.add_argument("--dia_num_heads", type=int, default=8)
    parser.add_argument("--dia_num_evidence_tokens", type=int, default=1)
    parser.add_argument("--dia_attn_dropout", type=float, default=0.0)
    parser.add_argument("--fusion_dropout", type=float, default=0.0)
    parser.add_argument(
        "--dia_fusion_mode",
        type=str,
        default="legacy",
        choices=[
            "legacy",
            "faithful_evidence_fusion",
            "decoupled_evidence_prompt",
            "sparse_dense",
            "bounded_sparse_dense",
            "latent_sparse_dense_dia",
            "evidence_feedback",
        ],
    )
    parser.add_argument("--token_bridge_init_gate", type=float, default=0.02)
    parser.add_argument("--dense_attn_clip", type=float, default=8.0)
    parser.add_argument("--role_adapter_hidden_dim", type=int, default=256)
    parser.add_argument("--role_max_delta_ratio", type=float, default=0.05)
    parser.add_argument("--dense_max_delta_ratio", type=float, default=0.10)
    parser.add_argument("--dense_confidence_power", type=float, default=0.5)
    parser.add_argument("--faithful_fusion_hidden_dim", type=int, default=256)
    parser.add_argument("--faithful_max_delta_ratio", type=float, default=0.15)
    parser.add_argument("--faithful_delta_gain", type=float, default=1.0)
    parser.add_argument("--latent_sparse_hidden_dim", type=int, default=256)
    parser.add_argument("--latent_sparse_max_delta_ratio", type=float, default=0.40)
    parser.add_argument("--latent_sparse_delta_gain", type=float, default=3.0)
    parser.add_argument("--latent_sparse_init_std", type=float, default=1e-3)
    parser.add_argument("--latent_dense_max_delta_ratio", type=float, default=0.15)
    parser.add_argument("--latent_dense_init_std", type=float, default=1e-3)
    parser.add_argument(
        "--visual_bottleneck_enabled",
        dest="visual_bottleneck_enabled",
        action="store_true",
        default=False,
        help="Enable CON-attention visual feature bottleneck before SAM mask decoder.",
    )
    parser.add_argument(
        "--no_visual_bottleneck",
        dest="visual_bottleneck_enabled",
        action="store_false",
        help="Disable the CON-attention visual feature bottleneck.",
    )
    parser.add_argument("--visual_bottleneck_beta", type=float, default=0.30)
    parser.add_argument("--visual_bottleneck_attn_clip", type=float, default=8.0)
    parser.add_argument(
        "--visual_bottleneck_max_delta_ratio",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--visual_bottleneck_confidence_power",
        type=float,
        default=0.25,
    )
    parser.add_argument("--visual_bottleneck_init_std", type=float, default=1e-3)
    parser.add_argument("--evidence_usage_loss_weight", type=float, default=0.10)
    parser.add_argument("--evidence_target_delta_ratio", type=float, default=0.12)
    parser.add_argument("--area_recall_loss_weight", type=float, default=0.20)
    parser.add_argument("--map_loss_weight", type=float, default=0.10)
    parser.add_argument("--anchor_loss_weight", type=float, default=0.10)
    parser.add_argument("--anchor_decay_steps", type=int, default=8000)
    parser.add_argument("--dia_loc_bias_init", type=float, default=-4.0)
    parser.add_argument("--dia_fusion_max_strength", type=float, default=0.15)
    parser.add_argument("--dia_fusion_warmup_steps", type=int, default=2000)
    parser.add_argument("--dia_fusion_ramp_steps", type=int, default=4000)
    parser.add_argument("--dia_gate_floor", type=float, default=0.10)
    parser.add_argument("--dia_init_gate", type=float, default=0.50)
    parser.add_argument("--dia_lr_multiplier", type=float, default=2.0)
    parser.add_argument(
        "--dia_training_stage",
        choices=["one_stage", "evidence", "fusion"],
        default="one_stage",
        help=(
            "one_stage keeps the existing DIA recipe. evidence trains CON "
            "attention/evidence first. fusion trains full DIA decoding after "
            "an evidence-stage checkpoint."
        ),
    )
    parser.add_argument("--dia_stage1_ce_loss_weight", type=float, default=0.25)
    parser.add_argument("--dia_stage1_attn_loss_weight", type=float, default=0.25)
    parser.add_argument(
        "--dia_stage1_evidence_usage_loss_weight",
        type=float,
        default=0.10,
    )
    parser.add_argument("--dia_stage1_area_recall_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--faithful_strict_config",
        action="store_true",
        default=False,
        help=(
            "Require the conservative faithful DIA recipe "
            "(attn_loss_weight=0.02, max_delta_ratio=0.15, delta_gain=1.0)."
        ),
    )
    parser.add_argument(
        "--allow_resume_hparam_mismatch",
        action="store_true",
        default=False,
        help=(
            "Allow resuming a DIA checkpoint while changing tunable strength "
            "hparams such as attention loss weight, residual cap, or delta gain."
        ),
    )
    parser.add_argument(
        "--dia_bypass_fusion",
        action="store_true",
        help="Debug only: send raw [SEG] projector output to SAM and bypass DIA fusion.",
    )
    parser.add_argument(
        "--init_con_from_seg",
        dest="init_con_from_seg",
        action="store_true",
    )
    parser.add_argument(
        "--no_init_con_from_seg",
        dest="init_con_from_seg",
        action="store_false",
    )
    parser.set_defaults(init_con_from_seg=True)

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    if args.explicit_con_in_conversation and not args.use_dia:
        raise ValueError("--explicit_con_in_conversation requires --use_dia.")
    if args.explicit_con_in_conversation and not args.init_con_from_seg:
        raise ValueError("--explicit_con_in_conversation requires --init_con_from_seg.")
    if args.dia_training_stage != "one_stage":
        if not args.use_dia:
            raise ValueError("--dia_training_stage requires --use_dia.")
        if args.dia_fusion_mode != "latent_sparse_dense_dia":
            raise ValueError(
                "Two-stage DIA is only defined for latent_sparse_dense_dia."
            )
    if args.dia_training_stage == "fusion" and args.resume and not args.resume_model_only:
        if args.local_rank == 0:
            print(
                "[WARN] dia_training_stage=fusion usually should use "
                "--resume_model_only when loading an evidence-stage checkpoint."
            )
    for name in (
        "dia_stage1_ce_loss_weight",
        "dia_stage1_attn_loss_weight",
        "dia_stage1_evidence_usage_loss_weight",
        "dia_stage1_area_recall_loss_weight",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    explicit_dia_modes = {
        "sparse_dense",
        "bounded_sparse_dense",
        "faithful_evidence_fusion",
        "decoupled_evidence_prompt",
        "evidence_feedback",
    }
    if args.dia_fusion_mode in explicit_dia_modes:
        if not args.use_dia:
            raise ValueError(f"{args.dia_fusion_mode} requires --use_dia.")
        if not args.explicit_con_in_conversation:
            raise ValueError(
                f"{args.dia_fusion_mode} requires --explicit_con_in_conversation."
            )
        if args.dia_bypass_fusion:
            raise ValueError(
                f"--dia_bypass_fusion is incompatible with {args.dia_fusion_mode}."
            )
    if args.dia_fusion_mode == "evidence_feedback":
        if not args.init_con_from_seg:
            raise ValueError("evidence_feedback requires --init_con_from_seg.")
        if args.dia_num_evidence_tokens != 1:
            raise ValueError("evidence_feedback requires --dia_num_evidence_tokens 1.")
        if args.dia_num_heads != 8:
            raise ValueError("evidence_feedback requires --dia_num_heads 8.")
        if abs(float(args.dia_attn_dropout)) > 1e-12:
            raise ValueError("evidence_feedback requires --dia_attn_dropout 0.0.")
        for name in (
            "map_loss_weight",
            "anchor_loss_weight",
            "dia_fusion_max_strength",
            "dia_gate_floor",
            "dia_init_gate",
            "dia_lr_multiplier",
        ):
            if getattr(args, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if not 0.0 < args.dia_fusion_max_strength <= 1.0:
            raise ValueError("dia_fusion_max_strength must be in (0, 1].")
        if args.dia_lr_multiplier <= 0.0:
            raise ValueError("dia_lr_multiplier must be positive.")
        if args.anchor_decay_steps < 0:
            raise ValueError("anchor_decay_steps must be non-negative.")
        if args.dia_fusion_warmup_steps < 0 or args.dia_fusion_ramp_steps <= 0:
            raise ValueError(
                "dia_fusion_warmup_steps must be >= 0 and "
                "dia_fusion_ramp_steps must be > 0."
            )
        if not 0.0 <= args.dia_gate_floor < args.dia_init_gate < 1.0:
            raise ValueError("expected 0 <= dia_gate_floor < dia_init_gate < 1.")
    if args.dia_fusion_mode == "faithful_evidence_fusion":
        if not args.init_con_from_seg:
            raise ValueError("faithful_evidence_fusion requires --init_con_from_seg.")
        if args.dia_num_evidence_tokens != 1:
            raise ValueError("faithful_evidence_fusion requires --dia_num_evidence_tokens 1.")
        if args.dia_num_heads != 8:
            raise ValueError("faithful_evidence_fusion requires --dia_num_heads 8.")
        if abs(float(args.dia_attn_dropout)) > 1e-12:
            raise ValueError("faithful_evidence_fusion requires --dia_attn_dropout 0.0.")
        if abs(float(args.fusion_dropout)) > 1e-12:
            raise ValueError("faithful_evidence_fusion requires --fusion_dropout 0.0.")
        if args.faithful_fusion_hidden_dim <= 0:
            raise ValueError("faithful_fusion_hidden_dim must be positive.")
        if args.faithful_max_delta_ratio <= 0.0:
            raise ValueError("faithful_max_delta_ratio must be positive.")
        if args.faithful_delta_gain <= 0.0:
            raise ValueError("faithful_delta_gain must be positive.")
        if args.faithful_strict_config:
            if abs(float(args.attn_loss_weight) - 0.02) > 1e-12:
                raise ValueError(
                    "strict faithful_evidence_fusion requires --attn_loss_weight 0.02."
                )
            if abs(float(args.faithful_max_delta_ratio) - 0.15) > 1e-12:
                raise ValueError(
                    "strict faithful_evidence_fusion requires "
                    "--faithful_max_delta_ratio 0.15."
                )
            if abs(float(args.faithful_delta_gain) - 1.0) > 1e-12:
                raise ValueError(
                    "strict faithful_evidence_fusion requires "
                    "--faithful_delta_gain 1.0."
                )
    if args.dia_fusion_mode == "sparse_dense":
        if not 0.0 < args.token_bridge_init_gate < 0.1:
            raise ValueError(
                "token_bridge_init_gate must be in (0, 0.1) for the "
                "baseline-anchored Explicit DIA experiment."
            )
    if args.dia_fusion_mode == "bounded_sparse_dense":
        if args.role_adapter_hidden_dim <= 0:
            raise ValueError("role_adapter_hidden_dim must be positive.")
        if not 0.0 < args.role_max_delta_ratio <= 0.10:
            raise ValueError("role_max_delta_ratio must be in (0, 0.10].")
        if not 0.0 < args.dense_max_delta_ratio <= 0.20:
            raise ValueError("dense_max_delta_ratio must be in (0, 0.20].")
        if args.dense_confidence_power <= 0.0:
            raise ValueError("dense_confidence_power must be positive.")
    if args.dia_fusion_mode == "latent_sparse_dense_dia":
        if not args.use_dia:
            raise ValueError("latent_sparse_dense_dia requires --use_dia.")
        if args.explicit_con_in_conversation:
            raise ValueError(
                "latent_sparse_dense_dia uses latent CON from [SEG] hidden; "
                "do not pass --explicit_con_in_conversation."
            )
        if args.dia_bypass_fusion:
            raise ValueError(
                "--dia_bypass_fusion is incompatible with latent_sparse_dense_dia."
            )
        if args.dia_num_evidence_tokens < 2:
            raise ValueError("latent_sparse_dense_dia requires --dia_num_evidence_tokens >= 2.")
        if args.dia_num_heads != 8:
            raise ValueError("latent_sparse_dense_dia requires --dia_num_heads 8.")
        if abs(float(args.dia_attn_dropout)) > 1e-12:
            raise ValueError("latent_sparse_dense_dia requires --dia_attn_dropout 0.0.")
        if args.latent_sparse_hidden_dim <= 0:
            raise ValueError("latent_sparse_hidden_dim must be positive.")
        if not 0.0 < args.latent_sparse_max_delta_ratio <= 0.60:
            raise ValueError("latent_sparse_max_delta_ratio must be in (0, 0.60].")
        if args.latent_sparse_delta_gain <= 0.0:
            raise ValueError("latent_sparse_delta_gain must be positive.")
        if args.latent_sparse_init_std < 0.0:
            raise ValueError("latent_sparse_init_std must be non-negative.")
        if not 0.0 < args.latent_dense_max_delta_ratio <= 0.20:
            raise ValueError("latent_dense_max_delta_ratio must be in (0, 0.20].")
        if args.latent_dense_init_std < 0.0:
            raise ValueError("latent_dense_init_std must be non-negative.")
        if args.visual_bottleneck_enabled:
            if not 0.0 < args.visual_bottleneck_beta <= 1.0:
                raise ValueError("visual_bottleneck_beta must be in (0, 1].")
            if args.visual_bottleneck_attn_clip <= 1.0:
                raise ValueError("visual_bottleneck_attn_clip must be > 1.")
            if not 0.0 < args.visual_bottleneck_max_delta_ratio <= 0.50:
                raise ValueError(
                    "visual_bottleneck_max_delta_ratio must be in (0, 0.50]."
                )
            if args.visual_bottleneck_confidence_power <= 0.0:
                raise ValueError("visual_bottleneck_confidence_power must be positive.")
            if args.visual_bottleneck_init_std < 0.0:
                raise ValueError("visual_bottleneck_init_std must be non-negative.")
        if args.evidence_usage_loss_weight < 0.0:
            raise ValueError("evidence_usage_loss_weight must be non-negative.")
        if not 0.0 <= args.evidence_target_delta_ratio <= args.latent_sparse_max_delta_ratio:
            raise ValueError(
                "evidence_target_delta_ratio must be between 0 and "
                "latent_sparse_max_delta_ratio."
            )
        if args.area_recall_loss_weight < 0.0:
            raise ValueError("area_recall_loss_weight must be non-negative.")
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)

    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        save_run_args(args)
        wandb.init(project="lisat", name=args.exp_name)
        print(
            "[Startup] "
            f"starting_from={args.version}, "
            f"manual_resume={args.resume or None}, "
            f"auto_resume={args.auto_resume}, "
            f"resume_model_only={args.resume_model_only}"
        )
        print(
            "[DIA] "
            f"enabled={args.use_dia}, "
            f"mode={args.dia_fusion_mode}, "
            f"stage={args.dia_training_stage}, "
            f"explicit_con={args.explicit_con_in_conversation}, "
            f"K={args.dia_num_evidence_tokens}, "
            f"heads={args.dia_num_heads}, "
            f"attn_dropout={args.dia_attn_dropout}, "
            f"attn_loss={args.attn_loss_weight}, "
            f"latent_sparse_cap={args.latent_sparse_max_delta_ratio}, "
            f"latent_sparse_gain={args.latent_sparse_delta_gain}, "
            f"latent_dense_cap={args.latent_dense_max_delta_ratio}, "
            f"veb={args.visual_bottleneck_enabled}, "
            f"veb_beta={args.visual_bottleneck_beta}, "
            f"veb_cap={args.visual_bottleneck_max_delta_ratio}, "
            f"veb_conf_power={args.visual_bottleneck_confidence_power}, "
            f"map_loss={args.map_loss_weight}, "
            f"anchor_loss={args.anchor_loss_weight}, "
            f"anchor_decay={args.anchor_decay_steps}, "
            f"loc_bias={args.dia_loc_bias_init}, "
            f"fusion_strength={args.dia_fusion_max_strength}, "
            f"fusion_warmup={args.dia_fusion_warmup_steps}, "
            f"fusion_ramp={args.dia_fusion_ramp_steps}, "
            f"dia_lr_multiplier={args.dia_lr_multiplier}"
        )

    # ---- Init conversation template ----
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # ---- Init model ----
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "use_dia": args.use_dia,
        "attn_loss_weight": args.attn_loss_weight if args.use_dia else 0.0,
        "dia_num_heads": args.dia_num_heads,
        "dia_num_evidence_tokens": args.dia_num_evidence_tokens,
        "dia_attn_dropout": args.dia_attn_dropout,
        "fusion_dropout": args.fusion_dropout,
        "dia_bypass_fusion": args.dia_bypass_fusion,
        "explicit_con_in_conversation": args.explicit_con_in_conversation,
        "dia_fusion_mode": args.dia_fusion_mode,
        "token_bridge_init_gate": args.token_bridge_init_gate,
        "dense_attn_clip": args.dense_attn_clip,
        "role_adapter_hidden_dim": args.role_adapter_hidden_dim,
        "role_max_delta_ratio": args.role_max_delta_ratio,
        "dense_max_delta_ratio": args.dense_max_delta_ratio,
        "dense_confidence_power": args.dense_confidence_power,
        "faithful_fusion_hidden_dim": args.faithful_fusion_hidden_dim,
        "faithful_max_delta_ratio": args.faithful_max_delta_ratio,
        "faithful_delta_gain": args.faithful_delta_gain,
        "faithful_strict_config": args.faithful_strict_config,
        "latent_sparse_hidden_dim": args.latent_sparse_hidden_dim,
        "latent_sparse_max_delta_ratio": args.latent_sparse_max_delta_ratio,
        "latent_sparse_delta_gain": args.latent_sparse_delta_gain,
        "latent_sparse_init_std": args.latent_sparse_init_std,
        "latent_dense_max_delta_ratio": args.latent_dense_max_delta_ratio,
        "latent_dense_init_std": args.latent_dense_init_std,
        "visual_bottleneck_enabled": args.visual_bottleneck_enabled,
        "visual_bottleneck_beta": args.visual_bottleneck_beta,
        "visual_bottleneck_attn_clip": args.visual_bottleneck_attn_clip,
        "visual_bottleneck_max_delta_ratio": (
            args.visual_bottleneck_max_delta_ratio
        ),
        "visual_bottleneck_confidence_power": (
            args.visual_bottleneck_confidence_power
        ),
        "visual_bottleneck_init_std": args.visual_bottleneck_init_std,
        "evidence_usage_loss_weight": (
            args.evidence_usage_loss_weight if args.use_dia else 0.0
        ),
        "evidence_target_delta_ratio": args.evidence_target_delta_ratio,
        "area_recall_loss_weight": (
            args.area_recall_loss_weight if args.use_dia else 0.0
        ),
        "map_loss_weight": args.map_loss_weight if args.use_dia else 0.0,
        "anchor_loss_weight": args.anchor_loss_weight if args.use_dia else 0.0,
        "anchor_decay_steps": args.anchor_decay_steps,
        "dia_loc_bias_init": args.dia_loc_bias_init,
        "dia_fusion_max_strength": args.dia_fusion_max_strength,
        "dia_fusion_warmup_steps": args.dia_fusion_warmup_steps,
        "dia_fusion_ramp_steps": args.dia_fusion_ramp_steps,
        "dia_gate_floor": args.dia_gate_floor,
        "dia_init_gate": args.dia_init_gate,
        "dia_training_stage": args.dia_training_stage,
        "dia_stage1_ce_loss_weight": args.dia_stage1_ce_loss_weight,
        "dia_stage1_attn_loss_weight": args.dia_stage1_attn_loss_weight,
        "dia_stage1_evidence_usage_loss_weight": (
            args.dia_stage1_evidence_usage_loss_weight
        ),
        "dia_stage1_area_recall_loss_weight": (
            args.dia_stage1_area_recall_loss_weight
        ),

    }
    tokenizer, model, vision_tower = init_LISAT_model(args, model_args)
    # from IPython import embed; embed()
    # Setup DDP
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # ---- Build training set ----
    train_dataset = HybridDataset(
        args.dataset_dir,
        vision_tower.image_processor,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        neg_refer_seg_data=args.neg_refer_seg_data,
        correct_refer_seg_data=args.correct_refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        geo_reason_seg_data=args.geo_reason_seg_data,
        dia_align_aug_source_dataset=args.dia_align_aug_source_dataset,
        dia_align_aug_source_rates=args.dia_align_aug_source_rates,
        dia_align_aug_positive_prob=args.dia_align_aug_positive_prob,
        dia_align_aug_background_scale=args.dia_align_aug_background_scale,
        dia_align_aug_mask_dilation=args.dia_align_aug_mask_dilation,
    )

    # ---- Build validation set ----
    train_dataset_names = args.dataset.split("||")
    if args.eval_dataset == "auto":
        args.eval_dataset = "refsegrs" if "refsegrs" in train_dataset_names else "geo_reason_seg"

    val_dataset = None
    if not args.no_eval or args.eval_only:
        if args.eval_dataset == "refsegrs":
            from dataloaders.refsegrs_dataset import RefSegRSDataset

            val_dataset = RefSegRSDataset(
                args.dataset_dir,
                vision_tower.image_processor,
                samples_per_epoch=args.eval_samples,
                image_size=args.image_size,
                num_classes_per_sample=1,
                split=args.eval_split,
                is_train=False,
            )
            print(
                f"Training with {len(train_dataset)} examples and validating "
                f"RefSegRS-{args.eval_split} with {len(val_dataset)} examples."
            )
        else:
            val_dataset = ReasonSegDataset(
                args.dataset_dir,
                vision_tower.image_processor,
                samples_per_epoch=args.eval_samples,
                image_size=args.image_size,
                num_classes_per_sample=3,
                reason_seg_data=f"GeoReasonSeg|{args.eval_split}",
                use_fp=False,
                is_train=False,
            )
            print(f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples.")
    else:
        print(f"Training with {len(train_dataset)} examples (no_eval).")

    # ---- Deepspeed Config ----
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
                "torch_adam": True,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {"enabled": args.precision == "fp16"},
        "bf16": {"enabled": args.precision == "bf16"},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": args.zero_stage,
            "contiguous_gradients": True,
            # Match the stable baseline recipe for mixed batches whose ranks can
            # exercise different optional branches, such as VQA vs segmentation.
            "overlap_comm": False,
            "ignore_unused_parameters": True,
            "reduce_scatter": True,
            "reduce_bucket_size": args.zero_bucket_size,
            "allgather_bucket_size": args.zero_bucket_size,
        },
    }
    if args.zero_stage == 3:
        ds_config["zero_optimization"].update(
            {
                "stage3_prefetch_bucket_size": args.zero_bucket_size,
                "stage3_param_persistence_threshold": 1e5,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True,
            }
        )
    if args.local_rank == 0:
        print(f"[DeepSpeed] zero_stage={args.zero_stage}, zero_bucket_size={args.zero_bucket_size}")

    model_parameters = model.parameters()
    if args.use_dia and args.dia_fusion_mode == "evidence_feedback":
        model_parameters = build_optimizer_param_groups(
            model,
            args.lr,
            args.dia_lr_multiplier,
        )
        if len(model_parameters) == 2:
            ds_config["scheduler"]["params"]["warmup_min_lr"] = [0.0, 0.0]
            ds_config["scheduler"]["params"]["warmup_max_lr"] = [
                args.lr,
                args.lr * args.dia_lr_multiplier,
            ]
        if args.local_rank == 0:
            group_lrs = [group["lr"] for group in model_parameters]
            group_sizes = [
                sum(param.numel() for param in group["params"])
                for group in model_parameters
            ]
            print(
                "[OptimizerGroups] "
                f"count={len(model_parameters)}, "
                f"lrs={group_lrs}, "
                f"sizes={group_sizes}"
            )

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn_train,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            explicit_con_in_conversation=args.explicit_con_in_conversation,
        ),
        config=ds_config,
    )

    # ---- Resume if needed ----
    if args.auto_resume and len(args.resume) == 0:
        maybe_resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(maybe_resume):
            args.resume = maybe_resume
            if args.local_rank == 0:
                print(f"[Resume] auto-resume found checkpoint: {args.resume}")
        elif args.local_rank == 0:
            print("[Resume] auto_resume=True but no ckpt_model was found.")
    elif args.local_rank == 0:
        print(
            "[Resume] "
            f"auto_resume={args.auto_resume}, manual_resume={args.resume or None}, "
            f"model_only={args.resume_model_only}"
        )

    if args.resume:
        _check_resume_mode_compatibility(args)
        if args.resume_model_only:
            load_kwargs = {
                "load_optimizer_states": False,
                "load_lr_scheduler_states": False,
                "load_module_only": True,
            }
            try:
                load_path, client_state = model_engine.load_checkpoint(
                    args.resume,
                    **load_kwargs,
                )
            except TypeError:
                load_kwargs.pop("load_module_only", None)
                load_path, client_state = model_engine.load_checkpoint(
                    args.resume,
                    **load_kwargs,
                )
            if args.local_rank == 0:
                print(
                    "[Resume] loaded model weights only from "
                    f"{load_path}; start_epoch remains {args.start_epoch}."
                )
        else:
            load_path, client_state = model_engine.load_checkpoint(args.resume)
            with open(os.path.join(args.resume, "latest"), "r") as f:
                ckpt_dir = f.readlines()[0].strip()
            args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
            print(f"[Resume] loaded_checkpoint={load_path}, start_epoch={args.start_epoch}")

    # ---- Validation DataLoader ----
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn_val,
                tokenizer=tokenizer,
                use_mm_start_end=args.use_mm_start_end,
                explicit_con_in_conversation=args.explicit_con_in_conversation,
            ),
        )

    train_iter = iter(train_loader)

    # Keep track of the best combined metric
    best_score = -1.0
    best_epoch = -1

    # ---- Evaluate-Only Mode ----
    if args.eval_only:
        if val_dataset is None:
            raise RuntimeError("Evaluation requested, but no validation dataset was built.")

        seg_metrics = validate_seg(val_loader, model_engine, 0, args)
        if args.eval_dataset == "refsegrs":
            metrics_record = {
                "phase": "eval_only",
                "dia_training_stage": args.dia_training_stage,
                "step": 0,
                "split": args.eval_split,
            }
            metrics_record.update(seg_metrics)
            append_metrics(
                args,
                metrics_record,
            )
            if args.local_rank == 0:
                print(
                    f"[Eval-Only: RefSegRS-{args.eval_split}] "
                    f"gIoU={seg_metrics['val_giou']:.4f}, "
                    f"cIoU={seg_metrics['val_ciou']:.4f}, "
                    f"mIoU={seg_metrics['val_miou']:.4f}, "
                    f"Acc={seg_metrics['val_acc']:.4f}"
                )
            return

        # Evaluate all VQA/Caption sets for the original GeoReasonSeg/VQA setup.
        nwpu_bleu4 = validate_vqa(
            args.vqa_eval_file_nwpu,
            model_engine,
            tokenizer,
            vision_tower,
            args.precision,
            args.image_size,
            args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            max_new_tokens=args.model_max_length,
        )
        sydney_bleu4 = validate_vqa(
            args.vqa_eval_file_sydney,
            model_engine,
            tokenizer,
            vision_tower,
            args.precision,
            args.image_size,
            args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            max_new_tokens=args.model_max_length,
        )
        ucm_bleu4 = validate_vqa(
            args.vqa_eval_file_ucm,
            model_engine,
            tokenizer,
            vision_tower,
            args.precision,
            args.image_size,
            args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            max_new_tokens=args.model_max_length,
        )

        combined_metric = (
            (nwpu_bleu4 / 65.8)
            + (sydney_bleu4 / 62.23)
            + (ucm_bleu4 / 72.34)
            + (seg_metrics["val_giou"] / 0.275)
        )

        metrics_record = {
            "phase": "eval_only",
            "dia_training_stage": args.dia_training_stage,
            "step": 0,
            "split": args.eval_dataset,
            "nwpu_bleu4": nwpu_bleu4,
            "sydney_bleu4": sydney_bleu4,
            "ucm_bleu4": ucm_bleu4,
            "best_score": combined_metric,
        }
        metrics_record.update(seg_metrics)
        append_metrics(
            args,
            metrics_record,
        )
        print(
            f"[Eval-Only] NWPU={nwpu_bleu4:.2f}, Sydney={sydney_bleu4:.2f}, "
            f"UCM={ucm_bleu4:.2f}, gIoU={seg_metrics['val_giou']:.4f}, "
            f"cIoU={seg_metrics['val_ciou']:.4f}, "
            f"mIoU={seg_metrics['val_miou']:.4f}, "
            f"Acc={seg_metrics['val_acc']:.4f}, "
            f"best_score={combined_metric:.4f}"
        )
        return

    # ---- Training Loop ----
    for epoch in range(args.start_epoch, args.epochs):
        # Train
        train_iter, global_iters = train_one_epoch(
            train_loader, model_engine, epoch, scheduler, train_iter, args
        )

        if not args.no_eval:
            seg_metrics = validate_seg(val_loader, model_engine, global_iters, args)

            if args.eval_dataset == "refsegrs":
                if args.local_rank == 0:
                    wandb.log(
                        seg_metrics,
                        step=global_iters,
                    )
            else:
                # Original GeoReasonSeg/VQA validation path.
                nwpu_bleu4 = validate_vqa(
                    args.vqa_eval_file_nwpu,
                    model_engine,
                    tokenizer,
                    vision_tower,
                    args.precision,
                    args.image_size,
                    args.conv_type,
                    use_mm_start_end=args.use_mm_start_end,
                    max_new_tokens=args.model_max_length,
                )
                sydney_bleu4 = validate_vqa(
                    args.vqa_eval_file_sydney,
                    model_engine,
                    tokenizer,
                    vision_tower,
                    args.precision,
                    args.image_size,
                    args.conv_type,
                    use_mm_start_end=args.use_mm_start_end,
                    max_new_tokens=args.model_max_length,
                )
                ucm_bleu4 = validate_vqa(
                    args.vqa_eval_file_ucm,
                    model_engine,
                    tokenizer,
                    vision_tower,
                    args.precision,
                    args.image_size,
                    args.conv_type,
                    use_mm_start_end=args.use_mm_start_end,
                    max_new_tokens=args.model_max_length,
                )

                if args.local_rank == 0:
                    wandb.log(
                        {
                            **seg_metrics,
                            "nwpu_bleu4": nwpu_bleu4,
                            "sydney_bleu4": sydney_bleu4,
                            "ucm_bleu4": ucm_bleu4,
                        },
                        step=global_iters,
                    )

            if args.best_metric == "combined" and args.eval_dataset != "refsegrs":
                combined_metric = (
                    (nwpu_bleu4 / 65.8)
                    + (sydney_bleu4 / 62.23)
                    + (ucm_bleu4 / 72.34)
                    + (seg_metrics["val_giou"] / 0.275)
                )
            elif args.best_metric == "combined":
                combined_metric = seg_metrics["val_giou"]
            else:
                combined_metric = seg_metrics[args.best_metric]

            is_best = combined_metric > best_score
            should_save_best = is_best and combined_metric > args.min_best_score_to_save
            best_score = max(best_score, combined_metric)
            if is_best:
                best_epoch = epoch

            metrics_record = {
                "phase": "val",
                "dia_training_stage": args.dia_training_stage,
                "epoch": epoch,
                "step": global_iters,
                "split": args.eval_split if args.eval_dataset == "refsegrs" else args.eval_dataset,
                "nwpu_bleu4": locals().get("nwpu_bleu4", ""),
                "sydney_bleu4": locals().get("sydney_bleu4", ""),
                "ucm_bleu4": locals().get("ucm_bleu4", ""),
                "best_metric": args.best_metric,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "is_best": is_best,
                "checkpoint_dir": os.path.join(args.log_dir, "best_ckpt_model") if should_save_best else "",
            }
            metrics_record.update(seg_metrics)
            append_metrics(
                args,
                metrics_record,
            )

            if args.local_rank == 0:
                wandb.log({"best_score": best_score}, step=global_iters)

            if should_save_best:
                save_best_checkpoint(
                    model_engine,
                    args,
                    epoch,
                    global_iters,
                    metrics_record,
                )
            elif is_best and args.local_rank == 0:
                print(
                    "[Best] metric improved but checkpoint save skipped because "
                    f"{args.best_metric}={combined_metric:.6g} <= "
                    f"min_best_score_to_save={args.min_best_score_to_save}."
                )

    if args.no_eval:
        save_dir = os.path.join(args.log_dir, "ckpt_model")
        if args.local_rank == 0:
            _remove_path(save_dir)
        _distributed_barrier()
        model_engine.save_checkpoint(save_dir)


def train_one_epoch(train_loader, model, epoch, scheduler, train_iter, args):
    """Main training loop (one epoch)."""
    keys = [
        "loss",
        "ce_loss",
        "mask_bce_loss",
        "mask_dice_loss",
        "mask_loss",
        "attn_alignment_loss",
        "evidence_map_loss",
        "anchor_loss",
        "anchor_loss_weight",
        "num_positive_masks",
        "num_valid_attn_masks",
        "num_valid_evidence_masks",
        "attention_entropy",
        "attention_normalized_entropy",
    ]
    if args.use_dia and args.dia_fusion_mode == "latent_sparse_dense_dia":
        keys.extend(
            [
                "latent_sparse_delta_ratio",
                "latent_dense_delta_ratio",
                "latent_dense_confidence",
                "visual_bottleneck_gate_mean",
                "visual_bottleneck_confidence",
                "visual_bottleneck_total_delta_ratio",
                "evidence_usage_loss",
                "area_recall_loss",
                "latent_con_count",
            ]
        )
    elif args.use_dia and args.dia_fusion_mode == "legacy":
        keys.extend(["res_scale", "gate_mean"])
    elif args.use_dia and args.dia_fusion_mode == "decoupled_evidence_prompt":
        keys.extend(["evidence_delta_ratio"])
    elif args.use_dia and args.dia_fusion_mode == "faithful_evidence_fusion":
        keys.extend(["evidence_delta_ratio", "faithful_smooth_scale"])
    elif args.use_dia and args.dia_fusion_mode == "evidence_feedback":
        keys.extend(
            [
                "map_prob_mean",
                "map_prob_max",
                "fusion_strength",
                "evidence_delta_ratio",
            ]
        )
    elif args.use_dia:
        keys.extend(["token_delta_ratio", "bounded_dense_delta_ratio", "dense_confidence"])
    loss_meters = {k: AverageMeter(k, ":.4f") for k in keys}

    progress = ProgressMeter(
        args.steps_per_epoch,
        list(loss_meters.values()),
        prefix=f"Epoch: [{epoch}]"
    )

    model.train()
    valid_attention_keys = {
        "bounded_dense_preclip_ratio",
        "bounded_dense_delta_ratio",
        "bounded_dense_bound_scale",
        "bounded_dense_bound_hit_rate",
        "dense_confidence",
        "dense_normalized_entropy",
        "dense_relative_attention_abs_mean",
        "faithful_raw_delta_ratio",
        "evidence_delta_ratio",
        "faithful_smooth_scale",
        "latent_sparse_raw_delta_ratio",
        "latent_sparse_delta_ratio",
        "latent_sparse_bound_scale",
        "latent_sparse_bound_hit_rate",
        "latent_dense_preclip_ratio",
            "latent_dense_delta_ratio",
            "latent_dense_bound_scale",
            "latent_dense_bound_hit_rate",
            "latent_dense_confidence",
            "visual_bottleneck_gate_mean",
            "visual_bottleneck_confidence",
            "visual_bottleneck_image_delta_ratio",
            "visual_bottleneck_residual_ratio",
            "visual_bottleneck_total_delta_ratio",
            "visual_bottleneck_bound_scale",
            "visual_bottleneck_bound_hit_rate",
            "attention_normalized_entropy",
            "valid_attention_entropy",
            "map_prob_mean",
            "map_prob_max",
            "fusion_strength",
        }

    for global_step in range(args.steps_per_epoch):
        for micro_step in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            input_dict = prepare_input(input_dict, args.precision, is_cuda=True)
            dia_global_step = epoch * args.steps_per_epoch + global_step
            output_dict = model(
                dia_global_step=dia_global_step,
                **input_dict,
            )

            batch_size = input_dict["images"].size(0)
            num_valid_attn = output_dict["num_valid_attn_masks"].item()
            for k in keys:
                if k in valid_attention_keys:
                    if num_valid_attn > 0:
                        loss_meters[k].update(output_dict[k].item(), num_valid_attn)
                else:
                    loss_meters[k].update(output_dict[k].item(), batch_size)

            model.backward(output_dict["loss"])
            model.step()

        # Log + reset
        if global_step % args.print_freq == (args.print_freq - 1):
            if args.distributed:
                for k in keys:
                    loss_meters[k].all_reduce()

            total_steps = global_step + args.steps_per_epoch * epoch
            if args.local_rank == 0:
                progress.display(global_step + 1)
                for k in keys:
                    wandb.log({k: loss_meters[k].avg}, step=total_steps)
                curr_lr = scheduler.get_last_lr()[0]
                wandb.log({"lr": curr_lr}, step=total_steps)
                train_metrics = {
                    "phase": "train",
                    "dia_training_stage": args.dia_training_stage,
                    "epoch": epoch,
                    "step": total_steps,
                    "lr": curr_lr,
                }
                train_metrics.update({k: loss_meters[k].avg for k in keys})
                append_metrics(args, train_metrics)

            for k in keys:
                loss_meters[k].reset()

    return train_iter, (epoch + 1) * args.steps_per_epoch


@torch.no_grad()
def validate_seg(val_loader, model_engine, global_iters, args):
    """Validate segmentation and report paper-friendly binary mask metrics."""
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    target_meter = AverageMeter("Target", ":6.3f", Summary.SUM)
    output_meter = AverageMeter("Output", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    pr_iou_meter = AverageMeter("PrIoU", ":6.3f", Summary.SUM)
    gt_fg_meter = AverageMeter("GtFgFrac", ":6.3f", Summary.AVERAGE)
    pred_fg_meter = AverageMeter("PredFgFrac", ":6.3f", Summary.AVERAGE)
    empty_gt_meter = AverageMeter("EmptyGt", ":6.3f", Summary.AVERAGE)
    empty_pred_meter = AverageMeter("EmptyPred", ":6.3f", Summary.AVERAGE)
    logit_meter = AverageMeter("MaskLogit", ":6.3f", Summary.AVERAGE)
    iou_thresholds = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])

    model_engine.eval()
    eval_name = "Val-Seg"
    if getattr(args, "eval_dataset", "") == "refsegrs":
        eval_name = f"Val-RefSegRS: {args.eval_split}"

    vis_saved = 0
    for input_dict in tqdm.tqdm(val_loader, desc=eval_name):
        torch.cuda.empty_cache()
        input_dict = prepare_input(input_dict, args.precision, is_cuda=True)
        output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        pred_scores = pred_masks[0]
        output_list = (pred_scores > args.mask_threshold).int()
        vis_saved = _save_seg_visualizations(
            input_dict, masks_list, output_list, args, global_iters, vis_saved
        )

        intersection = torch.zeros(2, device=masks_list.device)
        union = torch.zeros(2, device=masks_list.device)
        target = torch.zeros(2, device=masks_list.device)
        output = torch.zeros(2, device=masks_list.device)
        acc_iou = torch.zeros(2, device=masks_list.device)
        pr_hits = torch.zeros(len(iou_thresholds), device=masks_list.device)
        thresholds = iou_thresholds.to(device=masks_list.device)
        gt_fg_frac = 0.0
        pred_fg_frac = 0.0
        empty_gt = 0.0
        empty_pred = 0.0
        logit_mean = 0.0
        for mask_i, output_i, score_i in zip(masks_list, output_list, pred_scores):
            inter_i, union_i, target_i = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            output_i_area = union_i + inter_i - target_i
            acc_iou_i = inter_i / (union_i + 1e-5)
            acc_iou_i[union_i == 0] += 1.0
            intersection += inter_i
            union += union_i
            target += target_i
            output += output_i_area
            acc_iou += acc_iou_i
            pr_hits += (acc_iou_i[1] >= thresholds).float()
            valid_i = mask_i != 255
            valid_pixels = valid_i.sum().item()
            if valid_pixels > 0:
                gt_pixels = (mask_i[valid_i] == 1).sum().item()
                pred_pixels = (output_i[valid_i] == 1).sum().item()
                gt_fg_frac += gt_pixels / valid_pixels
                pred_fg_frac += pred_pixels / valid_pixels
                empty_gt += float(gt_pixels == 0)
                empty_pred += float(pred_pixels == 0)
                logit_mean += score_i[valid_i].float().mean().item()

        intersection_meter.update(intersection.cpu().numpy())
        union_meter.update(union.cpu().numpy())
        target_meter.update(target.cpu().numpy())
        output_meter.update(output.cpu().numpy())
        acc_iou_meter.update((acc_iou / masks_list.shape[0]).cpu().numpy(), n=masks_list.shape[0])
        pr_iou_meter.update((pr_hits / masks_list.shape[0]).cpu().numpy(), n=masks_list.shape[0])
        gt_fg_meter.update(gt_fg_frac / masks_list.shape[0], n=masks_list.shape[0])
        pred_fg_meter.update(pred_fg_frac / masks_list.shape[0], n=masks_list.shape[0])
        empty_gt_meter.update(empty_gt / masks_list.shape[0], n=masks_list.shape[0])
        empty_pred_meter.update(empty_pred / masks_list.shape[0], n=masks_list.shape[0])
        logit_meter.update(logit_mean / masks_list.shape[0], n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    target_meter.all_reduce()
    output_meter.all_reduce()
    acc_iou_meter.all_reduce()
    pr_iou_meter.all_reduce()
    gt_fg_meter.all_reduce()
    pred_fg_meter.all_reduce()
    empty_gt_meter.all_reduce()
    empty_pred_meter.all_reduce()
    logit_meter.all_reduce()

    intersection_sum = np.asarray(intersection_meter.sum, dtype=np.float64)
    union_sum = np.asarray(union_meter.sum, dtype=np.float64)
    target_sum = np.asarray(target_meter.sum, dtype=np.float64)
    output_sum = np.asarray(output_meter.sum, dtype=np.float64)

    iou_class = intersection_sum / (union_sum + 1e-10)
    class_acc = intersection_sum / (target_sum + 1e-10)
    valid_iou = union_sum > 0
    valid_acc = target_sum > 0

    ciou = float(iou_class[1])
    giou = float(acc_iou_meter.avg[1])
    miou = _mean_valid(iou_class, valid_iou)
    acc = _safe_div(intersection_sum.sum(), target_sum.sum())
    macc = _mean_valid(class_acc, valid_acc)
    fg_acc = float(class_acc[1]) if target_sum[1] > 0 else 0.0

    tp = intersection_sum[1]
    pred_fg = output_sum[1]
    target_fg = target_sum[1]
    precision = _safe_div(tp, pred_fg)
    recall = _safe_div(tp, target_fg)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    pr_iou = np.asarray(pr_iou_meter.avg, dtype=np.float64)
    pred_gt_fg_ratio = _safe_div(float(pred_fg_meter.avg), float(gt_fg_meter.avg))

    metrics = {
        "val_giou": giou,
        "val_ciou": ciou,
        "val_miou": miou,
        "val_acc": acc,
        "val_macc": macc,
        "val_fg_acc": fg_acc,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "val_dice": f1,
        "val_iou_bg": float(iou_class[0]),
        "val_iou_fg": ciou,
        "val_pr_50": float(pr_iou[0]),
        "val_pr_60": float(pr_iou[1]),
        "val_pr_70": float(pr_iou[2]),
        "val_pr_80": float(pr_iou[3]),
        "val_pr_90": float(pr_iou[4]),
        "val_gt_fg_frac": float(gt_fg_meter.avg),
        "val_pred_fg_frac": float(pred_fg_meter.avg),
        "val_pred_gt_fg_ratio": pred_gt_fg_ratio,
        "val_empty_gt_rate": float(empty_gt_meter.avg),
        "val_empty_pred_rate": float(empty_pred_meter.avg),
        "val_logit_mean": float(logit_meter.avg),
        "val_mask_threshold": float(args.mask_threshold),
    }

    if args.local_rank == 0:
        print(
            f"[{eval_name}] "
            f"gIoU={metrics['val_giou']:.4f}, "
            f"cIoU={metrics['val_ciou']:.4f}, "
            f"mIoU={metrics['val_miou']:.4f}, "
            f"Acc={metrics['val_acc']:.4f}, "
            f"F1={metrics['val_f1']:.4f}, "
            f"pred_fg={metrics['val_pred_fg_frac']:.4f}, "
            f"gt_fg={metrics['val_gt_fg_frac']:.4f}"
        )
        if (
            metrics["val_gt_fg_frac"] > 0
            and (
                metrics["val_pred_fg_frac"] <= 1e-8
                or metrics["val_pred_gt_fg_ratio"] < 0.25
                or metrics["val_pred_gt_fg_ratio"] > 4.0
            )
        ):
            print(
                "[WARN] Validation foreground ratio is outside the stable band "
                f"at mask_threshold={args.mask_threshold}: "
                f"pred/gt={metrics['val_pred_gt_fg_ratio']:.4f}. "
                "Check evidence map learning, prompt strength, and mask logits."
            )

    model_engine.train()
    return metrics


@torch.no_grad()
def validate_vqa(
    vqa_file,
    model_engine,
    tokenizer,
    vision_tower,
    precision="bf16",
    image_size=1024,
    conv_type="llava_v1",
    use_mm_start_end=True,
    max_new_tokens=256,
):
    """Validate on a vqa dataset (NWPU, Sydney, UCM)."""
    if not os.path.exists(vqa_file):
        print(f"[WARN] Skip VQA validation because file is missing: {vqa_file}")
        return 0.0
    if Bleu is None:
        print(
            "[WARN] Skip VQA validation because pycocoevalcap is not installed: "
            f"{vqa_file}"
        )
        return 0.0
    device = next(model_engine.parameters()).device
    model_engine.eval()
    conversation_lib.default_conversation = conversation_lib.conv_templates[conv_type]
    tokenizer.padding_side = "left"
    img_processor = ImageProcessor(vision_tower.image_processor, image_size)

    predictions_dict = {}
    references_dict = {}

    with open(vqa_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm.tqdm(lines, desc=f"Val-VQA ({os.path.basename(vqa_file)})"):
        example = json.loads(line.strip())
        question_id = example["question_id"]
        question = example["text"]
        references = example["answer"]
        if not isinstance(references, list):
            references = [references]
        references_dict[question_id] = references

        base_dir = os.path.dirname(vqa_file) 
        image_path = os.path.join(base_dir, example["image"])
        raw_image = Image.open(image_path).convert("RGB")
        image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(
            image_path
        )

        conv = conversation_lib.default_conversation.copy()
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)  
        conversation_list = [conv.get_prompt()]

        if use_mm_start_end:
            conversation_list = replace_image_tokens(conversation_list)

        input_ids, _ = tokenize_and_pad(conversation_list, tokenizer, padding="left")

        input_dict = {
            "images_clip": torch.stack([image_clip], dim=0),
            "images": torch.stack([image], dim=0),
            "input_ids": input_ids,
            "sam_mask_shape_list": [sam_mask_shape],
        }
        input_dict = prepare_input(input_dict, precision, is_cuda=True)

        output_ids, pred_masks, object_presence = model_engine.module.evaluate(
            input_dict["images_clip"],
            input_dict["images"],
            input_dict["input_ids"],
            input_dict["sam_mask_shape_list"],
            max_new_tokens=max_new_tokens,
        )
        real_output_ids = output_ids[:, input_ids.shape[1] :]
        pred_text = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)[0]

        predictions_dict[question_id] = [pred_text]

    # Compute BLEU with pycocoevalcap
    bleu_scorer = Bleu(n=4)
    bleu_score, _ = bleu_scorer.compute_score(references_dict, predictions_dict)
    bleu4 = bleu_score[3] * 100.0

    if torch.distributed.get_rank() == 0:
        print(
            f"[Val-VQA: {os.path.basename(vqa_file)}] "
            f"BLEU1={bleu_score[0]*100:.2f}, BLEU2={bleu_score[1]*100:.2f}, "
            f"BLEU3={bleu_score[2]*100:.2f}, BLEU4={bleu4:.2f}"
        )

    return bleu4

if __name__ == "__main__":
    main(sys.argv[1:])
