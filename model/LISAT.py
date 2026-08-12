from json import decoder
import math
import token
from turtle import left
from typing import List

from unittest.mock import inplace
# from Documents.trae_projects.LISA.experiments.heatmap_attn import configure_matplotlib
# from Documents.trae_projects.LISA.experiments.sd_token_cross_attention_heatmap import save_attention_debug_images
# from Documents.trae_projects.LISAt.runs.lisat_refsegrs_pre.ckpt_model.zero_to_fp32 import device
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model
from model.llava.model import *
from model.llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from .llava.model.language_model.llava_llama import (
    LlavaConfig,
    LlavaLlamaForCausalLM,
    LlavaLlamaModel,
)
from .segment_anything import build_sam_vit_h

from .DIA_LISAt import (
    BoundedDenseEvidencePrompt,
    ContextEvidenceAdapter,
    DenseEvidencePrompt,
    DecoupledMaskPrompt,
    EvidenceGuideFusion,
    EvidenceGuideFusionV2,
    EvidenceVisualBottleneck,
    ExplicitRoleAdapter,
    ExplicitTokenBridge,
    FaithfulEvidenceFusion,
    LatentSparseEvidenceFusion,
    SharedEvidenceAdapter,
    attention_alignment_loss,
    evidence_map_loss,
    prompt_anchor_loss,
)

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,
    eps=1e-6,
):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def compute_dia_loss_components(
    ce_loss,
    pred_masks,
    gt_masks,
    attn_maps_list,
    bce_loss_weight,
    dice_loss_weight,
    attn_loss_weight,
    strict_prompt_alignment=False,
    dia_fusion_mode="legacy",
    map_loss_weight=0.0,
):
    mask_bce_sum = ce_loss.new_zeros(())
    mask_dice_sum = ce_loss.new_zeros(())
    attn_loss_sum = ce_loss.new_zeros(())
    evidence_map_loss_sum = ce_loss.new_zeros(())
    num_masks = 0
    num_attn_masks = 0
    num_evidence_masks = 0
    use_evidence_feedback = dia_fusion_mode == "evidence_feedback"

    for batch_idx in range(len(pred_masks)):
        gt_mask = gt_masks[batch_idx]
        pred_mask = pred_masks[batch_idx]
        num_gt = int(gt_mask.shape[0])
        if num_gt == 0:
            continue

        gt_mask = gt_mask.to(device=pred_mask.device)
        if pred_mask.shape != gt_mask.shape:
            raise RuntimeError(
                f"Mask shape mismatch: pred={pred_mask.shape}, gt={gt_mask.shape}"
            )

        # Keep BCE/Dice in fp32 even when the model runs in bf16/fp16. Remote
        # sensing targets are often tiny, and full-precision mask losses preserve
        # the small foreground gradients that teach the decoder to leave the
        # all-background solution.
        pred_mask_for_loss = pred_mask.float()
        gt_mask_for_loss = gt_mask.float()

        mask_bce_sum = mask_bce_sum + (
            sigmoid_ce_loss(pred_mask_for_loss, gt_mask_for_loss, num_masks=num_gt) * num_gt
        )
        mask_dice_sum = mask_dice_sum + (
            dice_loss(pred_mask_for_loss, gt_mask_for_loss, num_masks=num_gt) * num_gt
        )

        evidence_or_attn = attn_maps_list[batch_idx]
        if evidence_or_attn is not None and use_evidence_feedback:
            loc_logits = evidence_or_attn
            if int(loc_logits.shape[0]) != num_gt:
                raise RuntimeError(
                    "Evidence map/GT mask count mismatch: "
                    f"maps={loc_logits.shape[0]}, gt={num_gt}."
                )
            evidence_map_loss_sum = evidence_map_loss_sum + (
                evidence_map_loss(
                    loc_logits,
                    gt_mask.to(loc_logits.device),
                )
                * num_gt
            )
            num_evidence_masks += num_gt
        elif evidence_or_attn is not None:
            attn_maps = evidence_or_attn
            if strict_prompt_alignment:
                if int(attn_maps.shape[0]) != num_gt:
                    raise RuntimeError(
                        "Attention/GT mask count mismatch: "
                        f"attn={attn_maps.shape[0]}, gt={num_gt}."
                    )
                attn_count = num_gt
            else:
                attn_count = min(int(attn_maps.shape[0]), num_gt)
            if attn_count > 0:
                attn_loss_sum = attn_loss_sum + (
                    attention_alignment_loss(
                        attn_maps[:attn_count],
                        gt_mask[:attn_count].to(attn_maps.device),
                    )
                    * attn_count
                )
                num_attn_masks += attn_count
        num_masks += num_gt

    if num_masks > 0:
        mask_bce_loss = bce_loss_weight * mask_bce_sum / num_masks
        mask_dice_loss = dice_loss_weight * mask_dice_sum / num_masks
    else:
        mask_bce_loss = ce_loss.new_zeros(())
        mask_dice_loss = ce_loss.new_zeros(())

    if num_attn_masks > 0:
        attn_loss = attn_loss_sum / num_attn_masks
    else:
        attn_loss = ce_loss.new_zeros(())

    if num_evidence_masks > 0:
        evidence_map_loss_value = evidence_map_loss_sum / num_evidence_masks
    else:
        evidence_map_loss_value = ce_loss.new_zeros(())

    mask_loss = mask_bce_loss + mask_dice_loss
    if use_evidence_feedback:
        total_loss = ce_loss + mask_loss + map_loss_weight * evidence_map_loss_value
        reported_attn_loss = evidence_map_loss_value
        reported_attn_masks = num_evidence_masks
    else:
        total_loss = ce_loss + mask_loss + attn_loss_weight * attn_loss
        reported_attn_loss = attn_loss
        reported_attn_masks = num_attn_masks

    return {
        "loss": total_loss,
        "mask_bce_loss": mask_bce_loss,
        "mask_dice_loss": mask_dice_loss,
        "mask_loss": mask_loss,
        "attn_alignment_loss": reported_attn_loss,
        "attn_loss": reported_attn_loss,
        "evidence_map_loss": evidence_map_loss_value,
        "num_positive_masks": ce_loss.new_tensor(float(num_masks)),
        "num_valid_attn_masks": ce_loss.new_tensor(float(reported_attn_masks)),
        "num_valid_evidence_masks": ce_loss.new_tensor(float(num_evidence_masks)),
    }


class LisatMetaModel(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.use_dia = getattr(config, "use_dia", False)
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)

        self.lisat_modules_initialized = False

    def initialize_lisat_modules(self, config):
        if self.lisat_modules_initialized:
            raise RuntimeError("initialize_lisat_modules() must be called only once.")

        # Build SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]

        def build_text_project():
            return nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim),
                nn.Dropout(0.0),
            )

        # self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        # self.text_hidden_fcs.train()
        
        self.text_hidden_fcs = nn.ModuleList([build_text_project()])
        if self.use_dia:
            self.con_hidden_fcs = nn.ModuleList([build_text_project()])
            fusion_mode = getattr(config, "dia_fusion_mode", "legacy")
            if fusion_mode == "evidence_feedback":
                self.context_adapter = SharedEvidenceAdapter(
                    dim=out_dim,
                    num_heads=getattr(config, "dia_num_heads", 8),
                    num_evidence_tokens=1,
                    loc_bias_init=getattr(config, "dia_loc_bias_init", -4.0),
                )
                self.evidence_fusion = EvidenceGuideFusionV2(
                    dim=out_dim,
                    max_strength=getattr(config, "dia_fusion_max_strength", 0.15),
                    warmup_steps=getattr(config, "dia_fusion_warmup_steps", 2000),
                    ramp_steps=getattr(config, "dia_fusion_ramp_steps", 4000),
                    gate_floor=getattr(config, "dia_gate_floor", 0.10),
                    init_gate=getattr(config, "dia_init_gate", 0.50),
                )
            else:
                self.context_adapter = ContextEvidenceAdapter(
                    dim=out_dim,
                    num_heads=getattr(config, "dia_num_heads", 8),
                    num_evidence_tokens=getattr(config, "dia_num_evidence_tokens", 1),
                    dropout=getattr(config, "dia_attn_dropout", 0.0),
                )
            if fusion_mode == "legacy":
                self.evidence_fusion = EvidenceGuideFusion(
                    dim=out_dim,
                    dropout=getattr(config, "fusion_dropout", 0.0),
                )
            elif fusion_mode == "decoupled_evidence_prompt":
                self.decoupled_mask_prompt = DecoupledMaskPrompt(
                    dim=out_dim,
                    hidden_dim=getattr(config, "faithful_fusion_hidden_dim", out_dim),
                    max_delta_ratio=getattr(config, "faithful_max_delta_ratio", 0.15),
                )
            elif fusion_mode == "faithful_evidence_fusion":
                self.faithful_evidence_fusion = FaithfulEvidenceFusion(
                    dim=out_dim,
                    hidden_dim=getattr(
                        config,
                        "faithful_fusion_hidden_dim",
                        out_dim,
                    ),
                    max_delta_ratio=getattr(
                        config,
                        "faithful_max_delta_ratio",
                        0.15,
                    ),
                    delta_gain=getattr(
                        config,
                        "faithful_delta_gain",
                        1.0,
                    ),
                )
            elif fusion_mode == "sparse_dense":
                self.explicit_token_bridge = ExplicitTokenBridge(
                    dim=out_dim,
                    init_gate=getattr(config, "token_bridge_init_gate", 0.02),
                )
                self.dense_evidence_prompt = DenseEvidencePrompt(
                    dim=out_dim,
                    attn_clip=getattr(config, "dense_attn_clip", 8.0),
                )
            elif fusion_mode == "bounded_sparse_dense":
                self.explicit_role_adapter = ExplicitRoleAdapter(
                    dim=out_dim,
                    hidden_dim=getattr(config, "role_adapter_hidden_dim", out_dim),
                    max_delta_ratio=getattr(config, "role_max_delta_ratio", 0.05),
                )
                self.bounded_dense_evidence_prompt = BoundedDenseEvidencePrompt(
                    dim=out_dim,
                    attn_clip=getattr(config, "dense_attn_clip", 8.0),
                    max_delta_ratio=getattr(config, "dense_max_delta_ratio", 0.10),
                    confidence_power=getattr(config, "dense_confidence_power", 0.5),
                )
            elif fusion_mode == "latent_sparse_dense_dia":
                self.latent_sparse_fusion = LatentSparseEvidenceFusion(
                    dim=out_dim,
                    hidden_dim=getattr(
                        config,
                        "latent_sparse_hidden_dim",
                        out_dim,
                    ),
                    max_delta_ratio=getattr(
                        config,
                        "latent_sparse_max_delta_ratio",
                        0.40,
                    ),
                    delta_gain=getattr(
                        config,
                        "latent_sparse_delta_gain",
                        3.0,
                    ),
                    target_delta_ratio=getattr(
                        config,
                        "evidence_target_delta_ratio",
                        0.12,
                    ),
                    init_std=getattr(
                        config,
                        "latent_sparse_init_std",
                        1e-3,
                    ),
                )
                self.latent_dense_evidence_prompt = BoundedDenseEvidencePrompt(
                    dim=out_dim,
                    attn_clip=getattr(config, "dense_attn_clip", 8.0),
                    max_delta_ratio=getattr(
                        config,
                        "latent_dense_max_delta_ratio",
                        0.15,
                    ),
                    confidence_power=getattr(config, "dense_confidence_power", 0.25),
                    out_proj_init_std=getattr(
                        config,
                        "latent_dense_init_std",
                        1e-3,
                    ),
                )
                if getattr(config, "visual_bottleneck_enabled", False):
                    self.evidence_visual_bottleneck = EvidenceVisualBottleneck(
                        dim=out_dim,
                        beta=getattr(config, "visual_bottleneck_beta", 0.30),
                        attn_clip=getattr(
                            config,
                            "visual_bottleneck_attn_clip",
                            8.0,
                        ),
                        max_delta_ratio=getattr(
                            config,
                            "visual_bottleneck_max_delta_ratio",
                            0.20,
                        ),
                        confidence_power=getattr(
                            config,
                            "visual_bottleneck_confidence_power",
                            0.25,
                        ),
                        init_std=getattr(
                            config,
                            "visual_bottleneck_init_std",
                            1e-3,
                        ),
                    )
            elif fusion_mode == "evidence_feedback":
                pass
            else:
                raise ValueError(f"Unsupported DIA fusion mode: {fusion_mode}")
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        self._mark_lisat_modules_hf_initialized()
        self.lisat_modules_initialized = True

    @property
    def evidence_adapter(self):
        if not self.use_dia:
            raise AttributeError("DIA evidence adapter is disabled because use_dia=False.")
        return self.context_adapter

    def _mark_lisat_modules_hf_initialized(self):
        """Protect missing LISAt modules from Transformers re-initialization.

        LISAt_PRE does not contain SAM, [SEG] projector, or DIA weights. During
        from_pretrained(), Transformers initializes missing modules unless they
        are marked as already initialized. These modules are intentionally built
        by LISAt/SAM code, so they must be skipped by HF's generic initializer.
        """
        modules = [self.visual_model, self.text_hidden_fcs]
        if self.use_dia:
            modules.extend([self.con_hidden_fcs, self.context_adapter])
            fusion_mode = getattr(self.config, "dia_fusion_mode", "legacy")
            if fusion_mode == "legacy":
                modules.append(self.evidence_fusion)
            elif fusion_mode == "evidence_feedback":
                modules.append(self.evidence_fusion)
            elif fusion_mode == "decoupled_evidence_prompt":
                modules.append(self.decoupled_mask_prompt)
            elif fusion_mode == "faithful_evidence_fusion":
                modules.append(self.faithful_evidence_fusion)
            elif fusion_mode == "sparse_dense":
                modules.extend(
                    [
                        self.explicit_token_bridge,
                        self.dense_evidence_prompt,
                    ]
                )
            elif fusion_mode == "bounded_sparse_dense":
                modules.extend(
                    [
                        self.explicit_role_adapter,
                        self.bounded_dense_evidence_prompt,
                    ]
                )
            elif fusion_mode == "latent_sparse_dense_dia":
                modules.extend(
                    [
                        self.latent_sparse_fusion,
                        self.latent_dense_evidence_prompt,
                    ]
                )
                if hasattr(self, "evidence_visual_bottleneck"):
                    modules.append(self.evidence_visual_bottleneck)
            else:
                raise ValueError(f"Unsupported DIA fusion mode: {fusion_mode}")
        for module in modules:
            for submodule in module.modules():
                submodule._is_hf_initialized = True


class LisatModel(LisatMetaModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(LisatModel, self).__init__(config, **kwargs)
        # Instead of forcing use_cache=False, let us keep it True for generation
        self.config.use_cache = True

        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISATForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", 0.5)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", 2.0)

        self.use_dia = kwargs.pop("use_dia", getattr(config, "use_dia", False))
        config.use_dia = self.use_dia
        self.explicit_con_in_conversation = kwargs.pop(
            "explicit_con_in_conversation",
            getattr(config, "explicit_con_in_conversation", False),
        )
        if not self.use_dia:
            self.explicit_con_in_conversation = False
        config.explicit_con_in_conversation = self.explicit_con_in_conversation
        self.dia_fusion_mode = kwargs.pop(
            "dia_fusion_mode",
            getattr(config, "dia_fusion_mode", "legacy"),
        )
        self.con_token_idx = kwargs.pop("con_token_idx", None)
        if self.dia_fusion_mode not in {
            "legacy",
            "faithful_evidence_fusion",
            "decoupled_evidence_prompt",
            "sparse_dense",
            "bounded_sparse_dense",
            "latent_sparse_dense_dia",
            "evidence_feedback",
        }:
            raise ValueError(f"Unsupported dia_fusion_mode={self.dia_fusion_mode}.")
        if self.dia_fusion_mode in {
            "sparse_dense",
            "bounded_sparse_dense",
            "faithful_evidence_fusion",
            "decoupled_evidence_prompt",
            "evidence_feedback",
        }:
            if not self.use_dia:
                raise ValueError(f"{self.dia_fusion_mode} DIA requires use_dia=True.")
            if not self.explicit_con_in_conversation:
                raise ValueError(
                    f"{self.dia_fusion_mode} DIA requires explicit_con_in_conversation=True."
                )
            if self.con_token_idx is None:
                raise ValueError(f"{self.dia_fusion_mode} requires con_token_idx.")
        if self.dia_fusion_mode == "latent_sparse_dense_dia":
            if not self.use_dia:
                raise ValueError("latent_sparse_dense_dia requires use_dia=True.")
            if self.explicit_con_in_conversation:
                raise ValueError(
                    "latent_sparse_dense_dia uses an internal latent concept query; "
                    "do not enable explicit_con_in_conversation."
                )
            if self.con_token_idx is None:
                raise ValueError("latent_sparse_dense_dia requires con_token_idx.")
        config.dia_fusion_mode = self.dia_fusion_mode
        config.token_bridge_init_gate = kwargs.pop(
            "token_bridge_init_gate",
            getattr(config, "token_bridge_init_gate", 0.02),
        )
        config.dense_attn_clip = kwargs.pop(
            "dense_attn_clip",
            getattr(config, "dense_attn_clip", 8.0),
        )
        config.role_adapter_hidden_dim = kwargs.pop(
            "role_adapter_hidden_dim",
            getattr(config, "role_adapter_hidden_dim", 256),
        )
        config.role_max_delta_ratio = kwargs.pop(
            "role_max_delta_ratio",
            getattr(config, "role_max_delta_ratio", 0.05),
        )
        config.dense_max_delta_ratio = kwargs.pop(
            "dense_max_delta_ratio",
            getattr(config, "dense_max_delta_ratio", 0.10),
        )
        config.dense_confidence_power = kwargs.pop(
            "dense_confidence_power",
            getattr(config, "dense_confidence_power", 0.5),
        )
        config.faithful_fusion_hidden_dim = kwargs.pop(
            "faithful_fusion_hidden_dim",
            getattr(config, "faithful_fusion_hidden_dim", 256),
        )
        config.faithful_max_delta_ratio = kwargs.pop(
            "faithful_max_delta_ratio",
            getattr(config, "faithful_max_delta_ratio", 0.15),
        )
        config.faithful_delta_gain = kwargs.pop(
            "faithful_delta_gain",
            getattr(config, "faithful_delta_gain", 1.0),
        )
        config.faithful_strict_config = kwargs.pop(
            "faithful_strict_config",
            getattr(config, "faithful_strict_config", False),
        )
        config.latent_sparse_hidden_dim = kwargs.pop(
            "latent_sparse_hidden_dim",
            getattr(config, "latent_sparse_hidden_dim", 256),
        )
        config.latent_sparse_max_delta_ratio = kwargs.pop(
            "latent_sparse_max_delta_ratio",
            getattr(config, "latent_sparse_max_delta_ratio", 0.40),
        )
        config.latent_sparse_delta_gain = kwargs.pop(
            "latent_sparse_delta_gain",
            getattr(config, "latent_sparse_delta_gain", 3.0),
        )
        config.latent_sparse_init_std = kwargs.pop(
            "latent_sparse_init_std",
            getattr(config, "latent_sparse_init_std", 1e-3),
        )
        config.latent_dense_max_delta_ratio = kwargs.pop(
            "latent_dense_max_delta_ratio",
            getattr(config, "latent_dense_max_delta_ratio", 0.15),
        )
        config.latent_dense_init_std = kwargs.pop(
            "latent_dense_init_std",
            getattr(config, "latent_dense_init_std", 1e-3),
        )
        config.visual_bottleneck_enabled = kwargs.pop(
            "visual_bottleneck_enabled",
            getattr(config, "visual_bottleneck_enabled", False),
        )
        config.visual_bottleneck_beta = kwargs.pop(
            "visual_bottleneck_beta",
            getattr(config, "visual_bottleneck_beta", 0.30),
        )
        config.visual_bottleneck_attn_clip = kwargs.pop(
            "visual_bottleneck_attn_clip",
            getattr(config, "visual_bottleneck_attn_clip", 8.0),
        )
        config.visual_bottleneck_max_delta_ratio = kwargs.pop(
            "visual_bottleneck_max_delta_ratio",
            getattr(config, "visual_bottleneck_max_delta_ratio", 0.20),
        )
        config.visual_bottleneck_confidence_power = kwargs.pop(
            "visual_bottleneck_confidence_power",
            getattr(config, "visual_bottleneck_confidence_power", 0.25),
        )
        config.visual_bottleneck_init_std = kwargs.pop(
            "visual_bottleneck_init_std",
            getattr(config, "visual_bottleneck_init_std", 1e-3),
        )
        self.evidence_usage_loss_weight = kwargs.pop(
            "evidence_usage_loss_weight",
            getattr(config, "evidence_usage_loss_weight", 0.10),
        )
        self.evidence_target_delta_ratio = kwargs.pop(
            "evidence_target_delta_ratio",
            getattr(config, "evidence_target_delta_ratio", 0.12),
        )
        self.area_recall_loss_weight = kwargs.pop(
            "area_recall_loss_weight",
            getattr(config, "area_recall_loss_weight", 0.20),
        )
        self.dia_training_stage = kwargs.pop(
            "dia_training_stage",
            getattr(config, "dia_training_stage", "one_stage"),
        )
        self.dia_stage1_ce_loss_weight = kwargs.pop(
            "dia_stage1_ce_loss_weight",
            getattr(config, "dia_stage1_ce_loss_weight", 0.25),
        )
        self.dia_stage1_attn_loss_weight = kwargs.pop(
            "dia_stage1_attn_loss_weight",
            getattr(config, "dia_stage1_attn_loss_weight", 0.25),
        )
        self.dia_stage1_evidence_usage_loss_weight = kwargs.pop(
            "dia_stage1_evidence_usage_loss_weight",
            getattr(config, "dia_stage1_evidence_usage_loss_weight", 0.10),
        )
        self.dia_stage1_area_recall_loss_weight = kwargs.pop(
            "dia_stage1_area_recall_loss_weight",
            getattr(config, "dia_stage1_area_recall_loss_weight", 0.0),
        )
        self.dia_bypass_fusion = kwargs.pop(
            "dia_bypass_fusion",
            getattr(config, "dia_bypass_fusion", False),
        )
        if (
            self.dia_fusion_mode in {
                "sparse_dense",
                "bounded_sparse_dense",
                "faithful_evidence_fusion",
                "latent_sparse_dense_dia",
                "evidence_feedback",
            }
            and self.dia_bypass_fusion
        ):
            raise ValueError(
                f"--dia_bypass_fusion is incompatible with {self.dia_fusion_mode} DIA."
            )
        config.dia_bypass_fusion = self.dia_bypass_fusion
        self.attn_loss_weight = kwargs.pop(
            "attn_loss_weight",
            getattr(config, "attn_loss_weight", 0.02),
        )
        if not self.use_dia:
            self.attn_loss_weight = 0.0
            self.evidence_usage_loss_weight = 0.0
            self.area_recall_loss_weight = 0.0
        config.evidence_usage_loss_weight = self.evidence_usage_loss_weight
        config.evidence_target_delta_ratio = self.evidence_target_delta_ratio
        config.area_recall_loss_weight = self.area_recall_loss_weight
        config.dia_training_stage = self.dia_training_stage
        config.dia_stage1_ce_loss_weight = self.dia_stage1_ce_loss_weight
        config.dia_stage1_attn_loss_weight = self.dia_stage1_attn_loss_weight
        config.dia_stage1_evidence_usage_loss_weight = (
            self.dia_stage1_evidence_usage_loss_weight
        )
        config.dia_stage1_area_recall_loss_weight = (
            self.dia_stage1_area_recall_loss_weight
        )
        config.dia_num_heads = kwargs.pop("dia_num_heads", getattr(config, "dia_num_heads", 8))
        config.dia_num_evidence_tokens = kwargs.pop(
            "dia_num_evidence_tokens",
            getattr(config, "dia_num_evidence_tokens", 1),
        )
        config.dia_attn_dropout = kwargs.pop(
            "dia_attn_dropout",
            getattr(config, "dia_attn_dropout", 0.0),
        )
        config.fusion_dropout = kwargs.pop(
            "fusion_dropout",
            getattr(config, "fusion_dropout", 0.0),
        )
        config.attn_loss_weight = self.attn_loss_weight
        self.map_loss_weight = kwargs.pop(
            "map_loss_weight",
            getattr(config, "map_loss_weight", 0.10),
        )
        self.anchor_loss_weight = kwargs.pop(
            "anchor_loss_weight",
            getattr(config, "anchor_loss_weight", 0.10),
        )
        self.anchor_decay_steps = kwargs.pop(
            "anchor_decay_steps",
            getattr(config, "anchor_decay_steps", 8000),
        )
        config.map_loss_weight = self.map_loss_weight
        config.anchor_loss_weight = self.anchor_loss_weight
        config.anchor_decay_steps = self.anchor_decay_steps
        config.dia_loc_bias_init = kwargs.pop(
            "dia_loc_bias_init",
            getattr(config, "dia_loc_bias_init", -4.0),
        )
        config.dia_fusion_max_strength = kwargs.pop(
            "dia_fusion_max_strength",
            getattr(config, "dia_fusion_max_strength", 0.10),
        )
        config.dia_fusion_warmup_steps = kwargs.pop(
            "dia_fusion_warmup_steps",
            getattr(config, "dia_fusion_warmup_steps", 500),
        )
        config.dia_fusion_ramp_steps = kwargs.pop(
            "dia_fusion_ramp_steps",
            getattr(config, "dia_fusion_ramp_steps", 1000),
        )
        config.dia_gate_floor = kwargs.pop(
            "dia_gate_floor",
            getattr(config, "dia_gate_floor", 0.10),
        )
        config.dia_init_gate = kwargs.pop(
            "dia_init_gate",
            getattr(config, "dia_init_gate", 0.50),
        )
        if self.dia_fusion_mode == "faithful_evidence_fusion":
            if config.dia_num_evidence_tokens != 1:
                raise ValueError("faithful_evidence_fusion requires K=1.")
            if config.dia_num_heads != 8:
                raise ValueError("faithful_evidence_fusion requires dia_num_heads=8.")
            if abs(float(config.dia_attn_dropout)) > 1e-12:
                raise ValueError("faithful_evidence_fusion requires dia_attn_dropout=0.0.")
            if abs(float(config.fusion_dropout)) > 1e-12:
                raise ValueError("faithful_evidence_fusion requires fusion_dropout=0.0.")
            if config.faithful_delta_gain <= 0.0:
                raise ValueError("faithful_evidence_fusion requires faithful_delta_gain > 0.")
            if config.faithful_strict_config:
                if abs(float(config.attn_loss_weight) - 0.02) > 1e-12:
                    raise ValueError(
                        "strict faithful_evidence_fusion requires attn_loss_weight=0.02."
                    )
                if abs(float(config.faithful_max_delta_ratio) - 0.15) > 1e-12:
                    raise ValueError(
                        "strict faithful_evidence_fusion requires "
                        "faithful_max_delta_ratio=0.15."
                    )
                if abs(float(config.faithful_delta_gain) - 1.0) > 1e-12:
                    raise ValueError(
                        "strict faithful_evidence_fusion requires "
                        "faithful_delta_gain=1.0."
                    )
        if self.dia_fusion_mode == "evidence_feedback":
            if not self.use_dia:
                raise ValueError("evidence_feedback requires use_dia=True.")
            if not self.explicit_con_in_conversation:
                raise ValueError(
                    "evidence_feedback requires explicit_con_in_conversation=True."
                )
            if config.dia_num_evidence_tokens != 1:
                raise ValueError("evidence_feedback requires K=1.")
            if abs(float(config.dia_attn_dropout)) > 1e-12:
                raise ValueError("evidence_feedback requires dia_attn_dropout=0.0.")
            if self.map_loss_weight < 0.0:
                raise ValueError("map_loss_weight must be non-negative.")
            if self.anchor_loss_weight < 0.0:
                raise ValueError("anchor_loss_weight must be non-negative.")
            if int(self.anchor_decay_steps) < 0:
                raise ValueError("anchor_decay_steps must be non-negative.")
        if self.dia_fusion_mode == "latent_sparse_dense_dia":
            if config.dia_num_evidence_tokens < 2:
                raise ValueError("latent_sparse_dense_dia requires K >= 2.")
            if config.dia_num_heads != 8:
                raise ValueError("latent_sparse_dense_dia requires dia_num_heads=8.")
            if abs(float(config.dia_attn_dropout)) > 1e-12:
                raise ValueError("latent_sparse_dense_dia requires dia_attn_dropout=0.0.")
            if config.latent_sparse_hidden_dim <= 0:
                raise ValueError("latent_sparse_hidden_dim must be positive.")
            if not 0.0 < config.latent_sparse_max_delta_ratio <= 0.60:
                raise ValueError("latent_sparse_max_delta_ratio must be in (0, 0.60].")
            if config.latent_sparse_delta_gain <= 0.0:
                raise ValueError("latent_sparse_delta_gain must be positive.")
            if config.latent_sparse_init_std < 0.0:
                raise ValueError("latent_sparse_init_std must be non-negative.")
            if not 0.0 < config.latent_dense_max_delta_ratio <= 0.20:
                raise ValueError("latent_dense_max_delta_ratio must be in (0, 0.20].")
            if config.latent_dense_init_std < 0.0:
                raise ValueError("latent_dense_init_std must be non-negative.")
            if config.visual_bottleneck_enabled:
                if not 0.0 < config.visual_bottleneck_beta <= 1.0:
                    raise ValueError("visual_bottleneck_beta must be in (0, 1].")
                if config.visual_bottleneck_attn_clip <= 1.0:
                    raise ValueError("visual_bottleneck_attn_clip must be > 1.")
                if not 0.0 < config.visual_bottleneck_max_delta_ratio <= 0.50:
                    raise ValueError(
                        "visual_bottleneck_max_delta_ratio must be in (0, 0.50]."
                    )
                if config.visual_bottleneck_confidence_power <= 0.0:
                    raise ValueError(
                        "visual_bottleneck_confidence_power must be positive."
                    )
                if config.visual_bottleneck_init_std < 0.0:
                    raise ValueError("visual_bottleneck_init_std must be non-negative.")
            if self.evidence_usage_loss_weight < 0.0:
                raise ValueError("evidence_usage_loss_weight must be non-negative.")
            if not 0.0 <= self.evidence_target_delta_ratio <= config.latent_sparse_max_delta_ratio:
                raise ValueError(
                    "evidence_target_delta_ratio must be between 0 and "
                    "latent_sparse_max_delta_ratio."
                )
            if self.area_recall_loss_weight < 0.0:
                raise ValueError("area_recall_loss_weight must be non-negative.")
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", config.mm_vision_tower or "openai/clip-vit-large-patch14"
            )
        else:
            config.mm_vision_tower = config.vision_tower

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        # self.con_token_idx = kwargs.pop("con_token_idx")
        super(LISATForCausalLM, self).__init__(config)
        self.model = LisatModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        # Build SAM, [SEG] projector, and optional DIA modules after HF
        # post_init() so generic LLaMA initialization cannot overwrite them.
        # This is still part of model construction; from_pretrained() loads
        # checkpoint weights only after __init__ returns.
        self.model.initialize_lisat_modules(self.model.config)

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def _zero_anchor_for_modules(self, reference_loss, modules):
        """Attach zero-valued gradients for optional DIA branches.

        Full mixed training can put a VQA/negative-only sample on one rank
        while another rank receives a segmentation sample. The optional DIA
        modules then appear unused on one rank, which can desynchronize ZeRO
        gradient reductions. This term keeps the graph aligned without changing
        the numerical loss.
        """
        anchor = reference_loss.new_zeros(())
        for module in modules:
            for param in module.parameters():
                if param.requires_grad:
                    anchor = anchor + param.sum() * 0.0
        return anchor

    def segmentation_zero_anchor(self, reference_loss):
        modules = [
            self.model.text_hidden_fcs,
            self.model.visual_model.mask_decoder,
        ]
        if self.use_dia:
            modules.extend(
                [
                    self.model.con_hidden_fcs,
                    self.model.context_adapter,
                ]
            )
            if self.dia_fusion_mode == "legacy":
                modules.append(self.model.evidence_fusion)
            elif self.dia_fusion_mode == "evidence_feedback":
                modules.append(self.model.evidence_fusion)
            elif self.dia_fusion_mode == "decoupled_evidence_prompt":
                modules.append(self.model.decoupled_mask_prompt)
            elif self.dia_fusion_mode == "faithful_evidence_fusion":
                modules.append(self.model.faithful_evidence_fusion)
            elif self.dia_fusion_mode == "sparse_dense":
                modules.extend(
                    [
                        self.model.explicit_token_bridge,
                        self.model.dense_evidence_prompt,
                    ]
                )
            elif self.dia_fusion_mode == "bounded_sparse_dense":
                modules.extend(
                    [
                        self.model.explicit_role_adapter,
                        self.model.bounded_dense_evidence_prompt,
                    ]
                )
            elif self.dia_fusion_mode == "latent_sparse_dense_dia":
                modules.extend(
                    [
                        self.model.latent_sparse_fusion,
                        self.model.latent_dense_evidence_prompt,
                    ]
                )
            else:
                raise RuntimeError(f"Unsupported DIA fusion mode: {self.dia_fusion_mode}")
        return self._zero_anchor_for_modules(
            reference_loss,
            modules,
        )

    def con_projector_zero_anchor(self, reference_loss):
        if not self.use_dia:
            return reference_loss.new_zeros(())
        return self._zero_anchor_for_modules(
            reference_loss,
            [self.model.con_hidden_fcs],
        )

    def anchor_weight_at(self, global_step):
        if (
            not self.training
            or not self.use_dia
            or self.dia_fusion_mode != "evidence_feedback"
        ):
            return 0.0
        if global_step is None:
            raise RuntimeError("training evidence_feedback requires dia_global_step")
        if self.anchor_decay_steps <= 0:
            return 0.0
        progress = min(
            1.0,
            int(global_step) / float(max(1, int(self.anchor_decay_steps))),
        )
        return float(self.anchor_loss_weight) * (1.0 - progress)

    def validate_prompt_mask_counts(self, seg_embeddings, con_embeddings, masks_list):
        if len(seg_embeddings) != len(masks_list):
            raise RuntimeError("image-group count mismatch")
        if self.use_dia and (
            con_embeddings is None
            or len(con_embeddings) != len(seg_embeddings)
        ):
            raise RuntimeError("DIA CON/SEG group mismatch")

        for image_idx, (seg_i, gt_i) in enumerate(zip(seg_embeddings, masks_list)):
            num_seg = int(seg_i.shape[0])
            num_gt = int(gt_i.shape[0])
            if num_seg != num_gt:
                raise RuntimeError(
                    f"[SEG]-mask mismatch: image={image_idx}, "
                    f"seg={num_seg}, gt={num_gt}"
                )
            if self.use_dia:
                num_con = int(con_embeddings[image_idx].shape[0])
                if num_con != num_seg:
                    raise RuntimeError(
                        f"[CON]-[SEG] mismatch: image={image_idx}, "
                        f"con={num_con}, seg={num_seg}"
                    )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs

    # The hidden-state sequence is longer than input_ids because one <image> token is
    # replaced by many visual patch embeddings. Align special-token masks to that sequence.
    def build_shifted_token_mask(self, input_ids, token_idx, hidden_len):
        token_mask = input_ids[:, 1:] == token_idx
        left_pad = self.get_vision_tower().num_patches - 1
        token_mask = torch.cat([
            torch.zeros(token_mask.shape[0], left_pad, dtype=torch.bool, device=token_mask.device),
            token_mask,
        ], dim=1)

        if token_mask.shape[1] < hidden_len:
            right_pad = hidden_len - token_mask.shape[1]
            token_mask = F.pad(token_mask, (0, right_pad), value=False)
        
        return token_mask[:, :hidden_len]

    def validate_explicit_con_seg_pairs(self, input_ids):
        """Check raw token IDs before hidden-state shifting."""
        if self.con_token_idx is None:
            raise RuntimeError("Explicit DIA requires a valid [CON] token id.")

        for row_idx, row in enumerate(input_ids):
            con_pos = (row == self.con_token_idx).nonzero(as_tuple=False).flatten()
            seg_pos = (row == self.seg_token_idx).nonzero(as_tuple=False).flatten()

            if con_pos.numel() != seg_pos.numel():
                raise RuntimeError(
                    "Explicit DIA requires equal [CON]/[SEG] counts, "
                    f"row={row_idx}, con={con_pos.numel()}, seg={seg_pos.numel()}."
                )
            if con_pos.numel() == 0:
                continue
            if not torch.equal(seg_pos, con_pos + 1):
                raise RuntimeError(
                    "Explicit DIA requires every [SEG] to immediately follow [CON], "
                    f"row={row_idx}, con_pos={con_pos.tolist()}, seg_pos={seg_pos.tolist()}."
                )

    def build_dia_token_masks(self, input_ids, hidden_len):
        """Return separate hidden-state masks for explicit [SEG] and [CON] tokens."""
        seg_token_mask = self.build_shifted_token_mask(
            input_ids=input_ids,
            token_idx=self.seg_token_idx,
            hidden_len=hidden_len,
        )
        if not self.use_dia or not self.explicit_con_in_conversation:
            return seg_token_mask, seg_token_mask

        self.validate_explicit_con_seg_pairs(input_ids)
        con_token_mask = self.build_shifted_token_mask(
            input_ids=input_ids,
            token_idx=self.con_token_idx,
            hidden_len=hidden_len,
        )

        seg_counts = seg_token_mask.int().sum(-1)
        con_counts = con_token_mask.int().sum(-1)
        if not torch.equal(seg_counts, con_counts):
            raise RuntimeError(
                "Explicit DIA shifted mask counts differ: "
                f"con={con_counts.tolist()}, seg={seg_counts.tolist()}."
            )
        if int(seg_token_mask.sum().item()) > 0 and torch.equal(seg_token_mask, con_token_mask):
            raise RuntimeError(
                "Explicit DIA [CON] and [SEG] masks are identical; expected adjacent hidden states."
            )
        return seg_token_mask, con_token_mask

    
    def split_embeddings_by_offset(self, flat_embeddings, token_mask, offset):
        """
        Group flattened special-token embeddings back to image level.

        token_mask is conversation-level, while offset stores which conversations
        belong to each image.
        """
        token_counts = token_mask.int().sum(-1)
        token_offset = token_counts.cumsum(-1)
        token_offset = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=token_offset.device), token_offset],
            dim=0,
        )

        offset = offset.to(token_offset.device)
        token_offset = token_offset[offset]

        grouped = []
        for i in range(len(token_offset) - 1):
            start_i, end_i = token_offset[i], token_offset[i + 1]
            grouped.append(flat_embeddings[start_i:end_i])

        return grouped

    def generate_pred_masks(
        self,
        seg_embeddings,
        con_embeddings,
        image_embeddings,
        sam_mask_shape_list,
        anchor_embeddings=None,
        dia_global_step=None,
    ):
        """Generate predicted masks from SAM prompt embeddings.

        Baseline mode (use_dia=False) keeps the original LISAt behavior: a
        single projected [SEG] embedding is sent to the SAM mask decoder. DIA
        mode adds the [CON] evidence retrieval and zero-initialized fusion path.
        """
        multimask_output = False
        pred_masks = []
        attn_maps_list = []
        debug_stats = {
            "gate_means": [],
            "attention_entropies": [],
            "token_gate_means": [],
            "token_delta_ratios": [],
            "dense_delta_ratios": [],
            "role_preclip_ratios": [],
            "role_delta_ratios": [],
            "role_bound_scales": [],
            "role_bound_hit_rates": [],
            "bounded_dense_preclip_ratios": [],
            "bounded_dense_delta_ratios": [],
            "bounded_dense_bound_scales": [],
            "bounded_dense_bound_hit_rates": [],
            "dense_confidence_means": [],
            "dense_normalized_entropies": [],
            "dense_relative_attention_abs_means": [],
            "faithful_raw_delta_ratios": [],
            "evidence_delta_ratios": [],
            "faithful_smooth_scales": [],
            "map_prob_means": [],
            "map_prob_maxes": [],
            "fusion_strengths": [],
            "latent_sparse_raw_delta_ratios": [],
            "latent_sparse_delta_ratios": [],
            "latent_sparse_bound_scales": [],
            "latent_sparse_bound_hit_rates": [],
            "latent_dense_preclip_ratios": [],
            "latent_dense_delta_ratios": [],
            "latent_dense_bound_scales": [],
            "latent_dense_bound_hit_rates": [],
            "latent_dense_confidence_means": [],
            "visual_bottleneck_gate_means": [],
            "visual_bottleneck_confidence_means": [],
            "visual_bottleneck_image_delta_ratios": [],
            "visual_bottleneck_residual_ratios": [],
            "visual_bottleneck_total_delta_ratios": [],
            "visual_bottleneck_bound_scales": [],
            "visual_bottleneck_bound_hit_rates": [],
            "evidence_usage_losses": [],
            "attention_normalized_entropies": [],
            "sam_prompt_encoder_calls": [],
            "sam_mask_decoder_calls": [],
        }
        for i in range(len(seg_embeddings)):
            seg_i = seg_embeddings[i]
            con_i = con_embeddings[i] if self.use_dia and con_embeddings is not None else None
            anchor_i = (
                anchor_embeddings[i]
                if anchor_embeddings is not None
                else None
            )

            input_size, original_size = sam_mask_shape_list[i]
            input_size = (int(input_size[0]), int(input_size[1]))
            original_size = (int(original_size[0]), int(original_size[1]))
            if seg_i.shape[0] == 0:
                pred_masks.append(
                    torch.empty(
                        0,
                        *original_size,
                        dtype=image_embeddings.dtype,
                        device=image_embeddings.device,
                    )
                )
                attn_maps_list.append(None)
                continue

            if self.use_dia:
                if con_i is None:
                    raise RuntimeError("DIA mask decoding requires concept embeddings.")
                if self.dia_fusion_mode == "evidence_feedback":
                    if con_i.shape[0] != seg_i.shape[0]:
                        raise RuntimeError(
                            "evidence_feedback requires one CON per SEG: "
                            f"image={i}, con={con_i.shape[0]}, seg={seg_i.shape[0]}."
                        )
                    num_prompts = seg_i.shape[0]
                elif self.dia_fusion_mode in {
                    "faithful_evidence_fusion", "decoupled_evidence_prompt"
                }:
                    if con_i.shape[0] != seg_i.shape[0]:
                        raise RuntimeError(
                            f"{self.dia_fusion_mode} requires one CON per SEG: "
                            f"image={i}, con={con_i.shape[0]}, seg={seg_i.shape[0]}."
                        )
                    num_prompts = seg_i.shape[0]
                elif self.dia_fusion_mode in {"sparse_dense", "bounded_sparse_dense"}:
                    if anchor_i is None:
                        raise RuntimeError(
                            f"{self.dia_fusion_mode} Explicit DIA requires baseline anchor embeddings."
                        )
                    if not (
                        anchor_i.shape[0] == seg_i.shape[0] == con_i.shape[0]
                    ):
                        raise RuntimeError(
                            f"{self.dia_fusion_mode} prompt counts differ: "
                            f"image={i}, anchor={anchor_i.shape[0]}, "
                            f"seg={seg_i.shape[0]}, con={con_i.shape[0]}."
                        )
                    num_prompts = seg_i.shape[0]
                elif self.dia_fusion_mode == "latent_sparse_dense_dia":
                    if con_i.shape[0] != seg_i.shape[0]:
                        raise RuntimeError(
                            "latent_sparse_dense_dia requires one latent CON per SEG: "
                            f"image={i}, con={con_i.shape[0]}, seg={seg_i.shape[0]}."
                        )
                    num_prompts = seg_i.shape[0]
                elif self.explicit_con_in_conversation:
                    if con_i.shape[0] != seg_i.shape[0]:
                        raise RuntimeError(
                            "Explicit DIA requires one [CON] embedding for every [SEG] prompt, "
                            f"image={i}, con={con_i.shape[0]}, seg={seg_i.shape[0]}."
                        )
                    num_prompts = seg_i.shape[0]
                else:
                    # Structural DIA compatibility: old data can contain only [SEG].
                    if con_i.shape[0] == 1 and seg_i.shape[0] > 1:
                        con_i = con_i.expand(seg_i.shape[0], -1)
                    if con_i.shape[0] == 0:
                        con_i = seg_i
                    num_prompts = min(seg_i.shape[0], con_i.shape[0])
                seg_i = seg_i[:num_prompts]
                con_i = con_i[:num_prompts]
                if anchor_i is not None:
                    anchor_i = anchor_i[:num_prompts]
            else:
                num_prompts = seg_i.shape[0]
                seg_i = seg_i[:num_prompts]

            decoder_dtype = next(self.model.visual_model.mask_decoder.parameters()).dtype
            decoder_device = image_embeddings.device

            seg_i = seg_i.to(device=decoder_device, dtype=decoder_dtype)
            if self.use_dia:
                con_i = con_i.to(device=decoder_device, dtype=decoder_dtype)
                if anchor_i is not None:
                    anchor_i = anchor_i.to(device=decoder_device, dtype=decoder_dtype)

            image_i = image_embeddings[i].unsqueeze(0).to(dtype=decoder_dtype)
            decoder_image_i = image_i
            image_pe = self.model.visual_model.prompt_encoder.get_dense_pe().to(
                device=decoder_device, dtype=decoder_dtype
            )

            if self.use_dia:
                if self.dia_fusion_mode == "evidence_feedback":
                    evidence_tokens, map_probs, loc_logits = self.model.context_adapter(
                        con_embeddings=con_i,
                        image_embeddings=image_i,
                        image_pe=image_pe,
                    )
                    attn_maps = loc_logits
                else:
                    evidence_tokens, attn_maps = self.model.context_adapter(
                        con_embeddings=con_i,
                        image_embeddings=image_i,
                        image_pe=image_pe,
                    )
                    map_probs = None
                dense_delta = None
                if self.dia_fusion_mode == "evidence_feedback":
                    if getattr(self, "dia_bypass_fusion", False):
                        prompt_tokens = seg_i.unsqueeze(1)
                    else:
                        prompt_tokens = self.model.evidence_fusion(
                            seg_embeddings=seg_i,
                            evidence_tokens=evidence_tokens,
                            global_step=dia_global_step,
                        )
                    fusion = self.model.evidence_fusion
                    gate_mean = getattr(fusion, "last_gate_mean", None)
                    delta_ratio = getattr(fusion, "last_delta_ratio", None)
                    fusion_strength = getattr(fusion, "last_strength", None)
                    if map_probs is not None:
                        debug_stats["map_prob_means"].append(
                            map_probs.detach().float().mean()
                        )
                        debug_stats["map_prob_maxes"].append(
                            map_probs.detach().float().amax()
                        )
                    if delta_ratio is not None:
                        debug_stats["evidence_delta_ratios"].append(
                            delta_ratio.float().mean()
                        )
                    if fusion_strength is not None:
                        debug_stats["fusion_strengths"].append(
                            fusion_strength.float().mean()
                        )
                elif self.dia_fusion_mode == "decoupled_evidence_prompt":
                    prompt_tokens = self.model.decoupled_mask_prompt(
                        evidence_tokens=evidence_tokens,
                        num_prompts=num_prompts,
                    )
                    ratio = self.model.decoupled_mask_prompt.last_delta_ratio
                    if ratio is not None:
                        debug_stats["evidence_delta_ratios"].append(
                            ratio.float().mean()
                        )
                    gate_mean = None
                elif self.dia_fusion_mode == "faithful_evidence_fusion":
                    prompt_tokens = self.model.faithful_evidence_fusion(
                        seg_embeddings=seg_i,
                        evidence_tokens=evidence_tokens,
                    )
                    fusion = self.model.faithful_evidence_fusion
                    faithful_stats = {
                        "faithful_raw_delta_ratios": getattr(
                            fusion,
                            "last_raw_delta_ratio",
                            None,
                        ),
                        "evidence_delta_ratios": getattr(
                            fusion,
                            "last_delta_ratio",
                            None,
                        ),
                        "faithful_smooth_scales": getattr(
                            fusion,
                            "last_smooth_scale",
                            None,
                        ),
                    }
                    for stat_name, stat_value in faithful_stats.items():
                        if stat_value is not None:
                            debug_stats[stat_name].append(stat_value.float().mean())
                    gate_mean = None
                elif self.dia_fusion_mode == "sparse_dense":
                    prompt_tokens = self.model.explicit_token_bridge(
                        anchor_embeddings=anchor_i,
                        explicit_embeddings=seg_i,
                    ).unsqueeze(1)
                    dense_delta = self.model.dense_evidence_prompt(
                        image_embeddings=image_i,
                        evidence_tokens=evidence_tokens,
                        attn_maps=attn_maps,
                    )
                    token_gate_mean = getattr(
                        self.model.explicit_token_bridge,
                        "last_gate_mean",
                        None,
                    )
                    token_delta_ratio = getattr(
                        self.model.explicit_token_bridge,
                        "last_delta_ratio",
                        None,
                    )
                    dense_delta_ratio = getattr(
                        self.model.dense_evidence_prompt,
                        "last_delta_ratio",
                        None,
                    )
                    if token_gate_mean is not None:
                        debug_stats["token_gate_means"].append(token_gate_mean)
                    if token_delta_ratio is not None:
                        debug_stats["token_delta_ratios"].append(token_delta_ratio)
                    if dense_delta_ratio is not None:
                        debug_stats["dense_delta_ratios"].append(dense_delta_ratio)
                    gate_mean = None
                elif self.dia_fusion_mode == "bounded_sparse_dense":
                    prompt_tokens = self.model.explicit_role_adapter(
                        anchor_embeddings=anchor_i,
                        explicit_embeddings=seg_i,
                    ).unsqueeze(1)
                    dense_delta = self.model.bounded_dense_evidence_prompt(
                        image_embeddings=image_i,
                        evidence_tokens=evidence_tokens,
                        attn_maps=attn_maps,
                    )
                    role = self.model.explicit_role_adapter
                    dense = self.model.bounded_dense_evidence_prompt
                    role_stats = {
                        "role_preclip_ratios": getattr(role, "last_preclip_ratio", None),
                        "role_delta_ratios": getattr(role, "last_delta_ratio", None),
                        "role_bound_scales": getattr(role, "last_bound_scale", None),
                        "role_bound_hit_rates": getattr(role, "last_bound_hit_rate", None),
                    }
                    dense_stats = {
                        "bounded_dense_preclip_ratios": getattr(dense, "last_preclip_ratio", None),
                        "bounded_dense_delta_ratios": getattr(dense, "last_delta_ratio", None),
                        "bounded_dense_bound_scales": getattr(dense, "last_bound_scale", None),
                        "bounded_dense_bound_hit_rates": getattr(dense, "last_bound_hit_rate", None),
                        "dense_confidence_means": getattr(dense, "last_confidence_mean", None),
                        "dense_normalized_entropies": getattr(dense, "last_normalized_entropy", None),
                        "dense_relative_attention_abs_means": getattr(
                            dense,
                            "last_relative_attention_abs_mean",
                            None,
                        ),
                    }
                    for stat_name, stat_value in {**role_stats, **dense_stats}.items():
                        if stat_value is not None:
                            debug_stats[stat_name].append(stat_value)
                    gate_mean = None
                elif self.dia_fusion_mode == "latent_sparse_dense_dia":
                    prompt_tokens = self.model.latent_sparse_fusion(
                        seg_embeddings=seg_i,
                        evidence_tokens=evidence_tokens,
                    )
                    dense_delta = self.model.latent_dense_evidence_prompt(
                        image_embeddings=image_i,
                        evidence_tokens=evidence_tokens,
                        attn_maps=attn_maps,
                    )
                    sparse = self.model.latent_sparse_fusion
                    dense = self.model.latent_dense_evidence_prompt
                    latent_stats = {
                        "latent_sparse_raw_delta_ratios": getattr(
                            sparse,
                            "last_raw_delta_ratio",
                            None,
                        ),
                        "latent_sparse_delta_ratios": getattr(
                            sparse,
                            "last_delta_ratio",
                            None,
                        ),
                        "latent_sparse_bound_scales": getattr(
                            sparse,
                            "last_bound_scale",
                            None,
                        ),
                        "latent_sparse_bound_hit_rates": getattr(
                            sparse,
                            "last_bound_hit_rate",
                            None,
                        ),
                        "latent_dense_preclip_ratios": getattr(
                            dense,
                            "last_preclip_ratio",
                            None,
                        ),
                        "latent_dense_delta_ratios": getattr(
                            dense,
                            "last_delta_ratio",
                            None,
                        ),
                        "latent_dense_bound_scales": getattr(
                            dense,
                            "last_bound_scale",
                            None,
                        ),
                        "latent_dense_bound_hit_rates": getattr(
                            dense,
                            "last_bound_hit_rate",
                            None,
                        ),
                        "latent_dense_confidence_means": getattr(
                            dense,
                            "last_confidence_mean",
                            None,
                        ),
                    }
                    for stat_name, stat_value in latent_stats.items():
                        if stat_value is not None:
                            debug_stats[stat_name].append(stat_value.float().mean())
                    usage_loss = getattr(sparse, "last_usage_loss", None)
                    if usage_loss is not None:
                        debug_stats["evidence_usage_losses"].append(usage_loss)
                    if hasattr(self.model, "evidence_visual_bottleneck"):
                        decoder_image_i = self.model.evidence_visual_bottleneck(
                            image_embeddings=image_i,
                            evidence_tokens=evidence_tokens,
                            attn_maps=attn_maps,
                        )
                        bottleneck = self.model.evidence_visual_bottleneck
                        bottleneck_stats = {
                            "visual_bottleneck_gate_means": getattr(
                                bottleneck,
                                "last_gate_mean",
                                None,
                            ),
                            "visual_bottleneck_confidence_means": getattr(
                                bottleneck,
                                "last_confidence_mean",
                                None,
                            ),
                            "visual_bottleneck_image_delta_ratios": getattr(
                                bottleneck,
                                "last_image_delta_ratio",
                                None,
                            ),
                            "visual_bottleneck_residual_ratios": getattr(
                                bottleneck,
                                "last_residual_delta_ratio",
                                None,
                            ),
                            "visual_bottleneck_total_delta_ratios": getattr(
                                bottleneck,
                                "last_total_delta_ratio",
                                None,
                            ),
                            "visual_bottleneck_bound_scales": getattr(
                                bottleneck,
                                "last_bound_scale",
                                None,
                            ),
                            "visual_bottleneck_bound_hit_rates": getattr(
                                bottleneck,
                                "last_bound_hit_rate",
                                None,
                            ),
                        }
                        for stat_name, stat_value in bottleneck_stats.items():
                            if stat_value is not None:
                                debug_stats[stat_name].append(
                                    stat_value.float().mean()
                                )
                    gate_mean = None
                elif getattr(self, "dia_bypass_fusion", False):
                    prompt_tokens = seg_i.unsqueeze(1)
                    gate_mean = None
                else:
                    prompt_tokens = self.model.evidence_fusion(
                        seg_embeddings=seg_i,
                        con_embeddings=con_i,
                        evidence_tokens=evidence_tokens,
                    )
                    gate_mean = getattr(self.model.evidence_fusion, "last_gate_mean", None)

                if gate_mean is not None:
                    debug_stats["gate_means"].append(gate_mean)
                if self.dia_fusion_mode == "evidence_feedback":
                    flat_attn = map_probs.detach().flatten(-2)
                    flat_attn = flat_attn / flat_attn.sum(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1e-8)
                    flat_attn = flat_attn.clamp_min(1e-8)
                else:
                    flat_attn = attn_maps.detach().flatten(-2).clamp_min(1e-8)
                entropy = -(flat_attn * flat_attn.log()).sum(dim=-1).mean()
                debug_stats["attention_entropies"].append(entropy)
                max_entropy = math.log(max(int(flat_attn.shape[-1]), 2))
                debug_stats["attention_normalized_entropies"].append(
                    (entropy / max_entropy).clamp(0.0, 1.0)
                )
            else:
                prompt_tokens = seg_i.unsqueeze(1)
                attn_maps = None

            debug_stats["sam_prompt_encoder_calls"].append(
                image_embeddings.new_tensor(1.0)
            )
            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=prompt_tokens,
            )
            # SAM's PromptEncoder creates an empty sparse prompt tensor in fp32.
            # Cast both prompt outputs back to the mask decoder dtype so bf16/fp16
            # training does not hit Float x BFloat16 linear layers.
            sparse_embeddings = sparse_embeddings.to(device=decoder_device, dtype=decoder_dtype)
            dense_embeddings = dense_embeddings.to(device=decoder_device, dtype=decoder_dtype)
            if self.use_dia and self.dia_fusion_mode in {
                "sparse_dense",
                "bounded_sparse_dense",
                "latent_sparse_dense_dia",
            }:
                dense_embeddings = dense_embeddings + dense_delta.to(
                    device=decoder_device,
                    dtype=decoder_dtype,
                )

            debug_stats["sam_mask_decoder_calls"].append(
                image_embeddings.new_tensor(1.0)
            )
            low_res_masks, _ = self.model.visual_model.mask_decoder(
                image_embeddings=decoder_image_i,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=input_size,
                original_size=original_size,
            )
            pred_masks.append(pred_mask[:, 0])
            attn_maps_list.append(attn_maps)
        return pred_masks, attn_maps_list, debug_stats


    def foreground_area_recall_loss(self, pred_masks, gt_masks, reference_loss):
        """Penalize DIA when predicted foreground area is below GT foreground area."""
        loss_sum = reference_loss.new_zeros(())
        num_masks = 0

        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            num_gt = int(gt_mask.shape[0])
            if num_gt == 0:
                continue
            if pred_mask.shape != gt_mask.shape:
                raise RuntimeError(
                    f"Mask shape mismatch for area recall: "
                    f"pred={pred_mask.shape}, gt={gt_mask.shape}"
                )

            gt_mask = gt_mask.to(device=pred_mask.device)
            valid = gt_mask.ne(255)
            valid_count = valid.flatten(1).float().sum(-1).clamp_min(1.0)
            pred_area = (
                pred_mask.float().sigmoid().flatten(1)
                * valid.flatten(1).float()
            ).sum(-1) / valid_count
            gt_area = (
                gt_mask.float().flatten(1)
                * valid.flatten(1).float()
            ).sum(-1) / valid_count
            loss_sum = loss_sum + F.relu(gt_area - pred_area).sum()
            num_masks += num_gt

        if num_masks == 0:
            return reference_loss.new_zeros(())
        return loss_sum / float(num_masks)


    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        sam_mask_shape_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        dia_global_step = kwargs.pop("dia_global_step", None)
        batch_size = len(sam_mask_shape_list)
        assert batch_size == len(offset) - 1

        # Inference or training?
        if inference:
            # Inference path
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                torch.cuda.empty_cache()
            output_hidden_states = output_i.hidden_states
            output = None
        else:
            # Training path
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        last_hidden_state = (
            output_hidden_states[-1]
            if isinstance(output_hidden_states, (list, tuple))
            else output_hidden_states
        )
        hidden_len = last_hidden_state.shape[1]

        if self.use_dia:
            seg_token_mask, con_token_mask = self.build_dia_token_masks(
                input_ids=input_ids,
                hidden_len=hidden_len,
            )
        else:
            seg_token_mask = self.build_shifted_token_mask(
                input_ids=input_ids,
                token_idx=self.seg_token_idx,
                hidden_len=hidden_len,
            )
            con_token_mask = None

        seg_hidden = last_hidden_state[seg_token_mask]
        seg_flat = self.model.text_hidden_fcs[0](seg_hidden)
        seg_embeddings = self.split_embeddings_by_offset(seg_flat, seg_token_mask, offset)
        anchor_embeddings = None
        if self.use_dia:
            con_hidden = last_hidden_state[con_token_mask]
            if con_hidden.shape[0] != seg_hidden.shape[0]:
                raise RuntimeError(
                    "DIA prompt/concept hidden counts differ before projection: "
                    f"con={con_hidden.shape[0]}, seg={seg_hidden.shape[0]}."
                )
            con_flat = self.model.con_hidden_fcs[0](con_hidden)
            con_embeddings = self.split_embeddings_by_offset(
                con_flat,
                con_token_mask,
                offset,
            )
            if self.dia_fusion_mode in {
                "sparse_dense",
                "bounded_sparse_dense",
                "evidence_feedback",
            }:
                anchor_flat = self.model.text_hidden_fcs[0](con_hidden)
                anchor_embeddings = self.split_embeddings_by_offset(
                    anchor_flat,
                    con_token_mask,
                    offset,
                )
        else:
            con_hidden = None
            con_embeddings = None

        if (
            self.use_dia
            and self.dia_fusion_mode == "evidence_feedback"
            and not inference
        ):
            self.validate_prompt_mask_counts(seg_embeddings, con_embeddings, masks_list)

        image_embeddings = self.get_visual_embs(images)

        # DIA-LISA mask decoding。
        pred_masks, attn_maps_list, dia_debug_stats = self.generate_pred_masks(
            seg_embeddings=seg_embeddings,
            con_embeddings=con_embeddings,
            image_embeddings=image_embeddings,
            sam_mask_shape_list=sam_mask_shape_list,
            anchor_embeddings=anchor_embeddings,
            dia_global_step=dia_global_step,
        )

        # If inference => return masks
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": masks_list,
            }

        # Otherwise, training => compute segmentation + LM losses
        model_output = output
        gt_masks = masks_list
        output_logits = model_output.logits
        if labels.ne(-100).any():
            ce_loss = model_output.loss * self.ce_loss_weight
        else:
            # Long language-only samples can lose all supervised tokens after
            # truncation. Keep the language branch in graph with zero loss.
            ce_loss = output_logits.sum() * 0.0

        has_positive_masks = any(int(gt_mask.shape[0]) > 0 for gt_mask in gt_masks)
        uses_con_projector = self.use_dia and any(
            con_i is not None and con_i.shape[0] > 0
            for con_i in con_embeddings
        )

        loss_dict = compute_dia_loss_components(
            ce_loss=ce_loss,
            pred_masks=pred_masks,
            gt_masks=gt_masks,
            attn_maps_list=attn_maps_list,
            bce_loss_weight=self.bce_loss_weight,
            dice_loss_weight=self.dice_loss_weight,
            attn_loss_weight=self.attn_loss_weight if self.use_dia else 0.0,
            strict_prompt_alignment=(
                self.use_dia
                and self.dia_fusion_mode in {
                    "faithful_evidence_fusion",
                    "decoupled_evidence_prompt",
                    "latent_sparse_dense_dia",
                    "evidence_feedback",
                }
            ),
            dia_fusion_mode=self.dia_fusion_mode if self.use_dia else "legacy",
            map_loss_weight=self.map_loss_weight if self.use_dia else 0.0,
        )

        graph_anchor = ce_loss.new_zeros(())
        if not has_positive_masks:
            graph_anchor = self.segmentation_zero_anchor(loss_dict["loss"])
        elif self.use_dia and not uses_con_projector:
            graph_anchor = self.con_projector_zero_anchor(loss_dict["loss"])

        def mean_or_zero(values, reference):
            if values:
                return torch.stack(values).mean().to(reference.device)
            return reference.new_zeros(())

        gate_mean = mean_or_zero(dia_debug_stats["gate_means"], ce_loss)
        attention_entropy = mean_or_zero(dia_debug_stats["attention_entropies"], ce_loss)
        token_gate_mean = mean_or_zero(dia_debug_stats["token_gate_means"], ce_loss)
        token_delta_ratio = mean_or_zero(dia_debug_stats["token_delta_ratios"], ce_loss)
        dense_delta_ratio = mean_or_zero(dia_debug_stats["dense_delta_ratios"], ce_loss)
        role_preclip_ratio = mean_or_zero(dia_debug_stats["role_preclip_ratios"], ce_loss)
        role_delta_ratio = mean_or_zero(dia_debug_stats["role_delta_ratios"], ce_loss)
        role_bound_scale = mean_or_zero(dia_debug_stats["role_bound_scales"], ce_loss)
        role_bound_hit_rate = mean_or_zero(dia_debug_stats["role_bound_hit_rates"], ce_loss)
        bounded_dense_preclip_ratio = mean_or_zero(
            dia_debug_stats["bounded_dense_preclip_ratios"],
            ce_loss,
        )
        bounded_dense_delta_ratio = mean_or_zero(
            dia_debug_stats["bounded_dense_delta_ratios"],
            ce_loss,
        )
        bounded_dense_bound_scale = mean_or_zero(
            dia_debug_stats["bounded_dense_bound_scales"],
            ce_loss,
        )
        bounded_dense_bound_hit_rate = mean_or_zero(
            dia_debug_stats["bounded_dense_bound_hit_rates"],
            ce_loss,
        )
        dense_confidence = mean_or_zero(
            dia_debug_stats["dense_confidence_means"],
            ce_loss,
        )
        dense_normalized_entropy = mean_or_zero(
            dia_debug_stats["dense_normalized_entropies"],
            ce_loss,
        )
        dense_relative_attention_abs_mean = mean_or_zero(
            dia_debug_stats["dense_relative_attention_abs_means"],
            ce_loss,
        )
        faithful_raw_delta_ratio = mean_or_zero(
            dia_debug_stats["faithful_raw_delta_ratios"],
            ce_loss,
        )
        evidence_delta_ratio = mean_or_zero(
            dia_debug_stats["evidence_delta_ratios"],
            ce_loss,
        )
        faithful_smooth_scale = mean_or_zero(
            dia_debug_stats["faithful_smooth_scales"],
            ce_loss,
        )
        latent_sparse_raw_delta_ratio = mean_or_zero(
            dia_debug_stats["latent_sparse_raw_delta_ratios"],
            ce_loss,
        )
        latent_sparse_delta_ratio = mean_or_zero(
            dia_debug_stats["latent_sparse_delta_ratios"],
            ce_loss,
        )
        latent_sparse_bound_scale = mean_or_zero(
            dia_debug_stats["latent_sparse_bound_scales"],
            ce_loss,
        )
        latent_sparse_bound_hit_rate = mean_or_zero(
            dia_debug_stats["latent_sparse_bound_hit_rates"],
            ce_loss,
        )
        latent_dense_preclip_ratio = mean_or_zero(
            dia_debug_stats["latent_dense_preclip_ratios"],
            ce_loss,
        )
        latent_dense_delta_ratio = mean_or_zero(
            dia_debug_stats["latent_dense_delta_ratios"],
            ce_loss,
        )
        latent_dense_bound_scale = mean_or_zero(
            dia_debug_stats["latent_dense_bound_scales"],
            ce_loss,
        )
        latent_dense_bound_hit_rate = mean_or_zero(
            dia_debug_stats["latent_dense_bound_hit_rates"],
            ce_loss,
        )
        latent_dense_confidence = mean_or_zero(
            dia_debug_stats["latent_dense_confidence_means"],
            ce_loss,
        )
        visual_bottleneck_gate_mean = mean_or_zero(
            dia_debug_stats["visual_bottleneck_gate_means"],
            ce_loss,
        )
        visual_bottleneck_confidence = mean_or_zero(
            dia_debug_stats["visual_bottleneck_confidence_means"],
            ce_loss,
        )
        visual_bottleneck_image_delta_ratio = mean_or_zero(
            dia_debug_stats["visual_bottleneck_image_delta_ratios"],
            ce_loss,
        )
        visual_bottleneck_residual_ratio = mean_or_zero(
            dia_debug_stats["visual_bottleneck_residual_ratios"],
            ce_loss,
        )
        visual_bottleneck_total_delta_ratio = mean_or_zero(
            dia_debug_stats["visual_bottleneck_total_delta_ratios"],
            ce_loss,
        )
        visual_bottleneck_bound_scale = mean_or_zero(
            dia_debug_stats["visual_bottleneck_bound_scales"],
            ce_loss,
        )
        visual_bottleneck_bound_hit_rate = mean_or_zero(
            dia_debug_stats["visual_bottleneck_bound_hit_rates"],
            ce_loss,
        )
        evidence_usage_loss = mean_or_zero(
            dia_debug_stats["evidence_usage_losses"],
            ce_loss,
        )
        map_prob_mean = mean_or_zero(
            dia_debug_stats["map_prob_means"],
            ce_loss,
        )
        map_prob_max = mean_or_zero(
            dia_debug_stats["map_prob_maxes"],
            ce_loss,
        )
        fusion_strength = mean_or_zero(
            dia_debug_stats["fusion_strengths"],
            ce_loss,
        )
        area_recall_loss = (
            self.foreground_area_recall_loss(
                pred_masks,
                gt_masks,
                ce_loss,
            )
            if self.use_dia and self.dia_fusion_mode == "latent_sparse_dense_dia"
            else ce_loss.new_zeros(())
        )
        anchor_loss = ce_loss.new_zeros(())
        anchor_loss_weight = ce_loss.new_zeros(())
        if (
            self.use_dia
            and self.dia_fusion_mode == "evidence_feedback"
            and anchor_embeddings is not None
        ):
            anchor_weight_value = self.anchor_weight_at(dia_global_step)
            anchor_loss_weight = ce_loss.new_tensor(anchor_weight_value)
            if anchor_weight_value > 0.0:
                anchor_loss = prompt_anchor_loss(seg_embeddings, anchor_embeddings)
        if self.use_dia and self.dia_fusion_mode == "latent_sparse_dense_dia":
            if self.training and self.dia_training_stage == "evidence":
                # Stage 1 teaches [CON] to retrieve mask-aligned visual evidence
                # before the SAM decoding path is allowed to dominate the loss.
                loss_dict["loss"] = (
                    self.dia_stage1_ce_loss_weight * ce_loss
                    + self.dia_stage1_attn_loss_weight
                    * loss_dict["attn_alignment_loss"]
                    + self.dia_stage1_evidence_usage_loss_weight
                    * evidence_usage_loss
                    + self.dia_stage1_area_recall_loss_weight
                    * area_recall_loss
                    + graph_anchor
                )
            else:
                loss_dict["loss"] = (
                    loss_dict["loss"]
                    + self.evidence_usage_loss_weight * evidence_usage_loss
                    + self.area_recall_loss_weight * area_recall_loss
                    + graph_anchor
                )
        elif self.use_dia and self.dia_fusion_mode == "evidence_feedback":
            loss_dict["loss"] = (
                loss_dict["loss"]
                + anchor_loss_weight * anchor_loss
                + graph_anchor
            )
        else:
            loss_dict["loss"] = (
                loss_dict["loss"]
                + graph_anchor
            )
        attention_normalized_entropy = mean_or_zero(
            dia_debug_stats["attention_normalized_entropies"],
            ce_loss,
        )
        sam_prompt_encoder_calls = mean_or_zero(
            dia_debug_stats["sam_prompt_encoder_calls"],
            ce_loss,
        )
        sam_mask_decoder_calls = mean_or_zero(
            dia_debug_stats["sam_mask_decoder_calls"],
            ce_loss,
        )
        if self.use_dia and self.dia_fusion_mode == "legacy":
            res_scale = torch.tanh(self.model.evidence_fusion.res_scale.detach()).to(ce_loss.device)
        else:
            res_scale = ce_loss.new_zeros(())

        explicit_con_count = ce_loss.new_zeros(())
        explicit_seg_count = ce_loss.new_zeros(())
        explicit_paired_count = ce_loss.new_zeros(())
        explicit_orphan_con_count = ce_loss.new_zeros(())
        explicit_orphan_seg_count = ce_loss.new_zeros(())
        explicit_invalid_row_count = ce_loss.new_zeros(())
        explicit_pair_rate = ce_loss.new_zeros(())
        explicit_hidden_cosine = ce_loss.new_zeros(())
        latent_con_count = ce_loss.new_zeros(())
        if self.use_dia and self.explicit_con_in_conversation:
            explicit_con_count = con_token_mask.sum().to(
                device=ce_loss.device,
                dtype=ce_loss.dtype,
            )
            explicit_seg_count = seg_token_mask.sum().to(
                device=ce_loss.device,
                dtype=ce_loss.dtype,
            )
            if self.explicit_con_in_conversation:
                # Reaching this point means raw tokens and shifted masks passed validation.
                explicit_paired_count = explicit_seg_count
                explicit_pair_rate = ce_loss.new_ones(())
            if (
                con_hidden is not None
                and con_hidden.shape[0] > 0
                and con_hidden.shape == seg_hidden.shape
            ):
                explicit_hidden_cosine = F.cosine_similarity(
                    con_hidden.detach().float(),
                    seg_hidden.detach().float(),
                    dim=-1,
                ).mean().to(device=ce_loss.device, dtype=ce_loss.dtype)
        elif self.use_dia and self.dia_fusion_mode == "latent_sparse_dense_dia":
            latent_con_count = seg_token_mask.sum().to(
                device=ce_loss.device,
                dtype=ce_loss.dtype,
            )
            explicit_seg_count = latent_con_count

        loss_dict["ce_loss"] = ce_loss
        loss_dict["evidence_usage_loss"] = evidence_usage_loss.detach()
        loss_dict["area_recall_loss"] = area_recall_loss.detach()
        loss_dict["anchor_loss"] = anchor_loss.detach()
        loss_dict["anchor_loss_weight"] = anchor_loss_weight.detach()
        loss_dict["map_prob_mean"] = map_prob_mean.detach()
        loss_dict["map_prob_max"] = map_prob_max.detach()
        loss_dict["fusion_strength"] = fusion_strength.detach()
        loss_dict["res_scale"] = res_scale
        loss_dict["gate_mean"] = gate_mean.detach()
        loss_dict["attention_entropy"] = attention_entropy.detach()
        loss_dict["valid_attention_entropy"] = attention_entropy.detach()
        loss_dict["token_gate_mean"] = token_gate_mean.detach()
        loss_dict["token_delta_ratio"] = token_delta_ratio.detach()
        loss_dict["dense_delta_ratio"] = dense_delta_ratio.detach()
        loss_dict["role_preclip_ratio"] = role_preclip_ratio.detach()
        loss_dict["role_delta_ratio"] = role_delta_ratio.detach()
        loss_dict["role_bound_scale"] = role_bound_scale.detach()
        loss_dict["role_bound_hit_rate"] = role_bound_hit_rate.detach()
        loss_dict["bounded_dense_preclip_ratio"] = bounded_dense_preclip_ratio.detach()
        loss_dict["bounded_dense_delta_ratio"] = bounded_dense_delta_ratio.detach()
        loss_dict["bounded_dense_bound_scale"] = bounded_dense_bound_scale.detach()
        loss_dict["bounded_dense_bound_hit_rate"] = bounded_dense_bound_hit_rate.detach()
        loss_dict["dense_confidence"] = dense_confidence.detach()
        loss_dict["dense_normalized_entropy"] = dense_normalized_entropy.detach()
        loss_dict["dense_relative_attention_abs_mean"] = (
            dense_relative_attention_abs_mean.detach()
        )
        loss_dict["faithful_raw_delta_ratio"] = faithful_raw_delta_ratio.detach()
        loss_dict["evidence_delta_ratio"] = evidence_delta_ratio.detach()
        loss_dict["faithful_smooth_scale"] = faithful_smooth_scale.detach()
        loss_dict["latent_sparse_raw_delta_ratio"] = (
            latent_sparse_raw_delta_ratio.detach()
        )
        loss_dict["latent_sparse_delta_ratio"] = latent_sparse_delta_ratio.detach()
        loss_dict["latent_sparse_bound_scale"] = latent_sparse_bound_scale.detach()
        loss_dict["latent_sparse_bound_hit_rate"] = (
            latent_sparse_bound_hit_rate.detach()
        )
        loss_dict["latent_dense_preclip_ratio"] = latent_dense_preclip_ratio.detach()
        loss_dict["latent_dense_delta_ratio"] = latent_dense_delta_ratio.detach()
        loss_dict["latent_dense_bound_scale"] = latent_dense_bound_scale.detach()
        loss_dict["latent_dense_bound_hit_rate"] = (
            latent_dense_bound_hit_rate.detach()
        )
        loss_dict["latent_dense_confidence"] = latent_dense_confidence.detach()
        loss_dict["visual_bottleneck_gate_mean"] = (
            visual_bottleneck_gate_mean.detach()
        )
        loss_dict["visual_bottleneck_confidence"] = (
            visual_bottleneck_confidence.detach()
        )
        loss_dict["visual_bottleneck_image_delta_ratio"] = (
            visual_bottleneck_image_delta_ratio.detach()
        )
        loss_dict["visual_bottleneck_residual_ratio"] = (
            visual_bottleneck_residual_ratio.detach()
        )
        loss_dict["visual_bottleneck_total_delta_ratio"] = (
            visual_bottleneck_total_delta_ratio.detach()
        )
        loss_dict["visual_bottleneck_bound_scale"] = (
            visual_bottleneck_bound_scale.detach()
        )
        loss_dict["visual_bottleneck_bound_hit_rate"] = (
            visual_bottleneck_bound_hit_rate.detach()
        )
        loss_dict["attention_normalized_entropy"] = (
            attention_normalized_entropy.detach()
        )
        loss_dict["explicit_con_count"] = explicit_con_count.detach()
        loss_dict["explicit_seg_count"] = explicit_seg_count.detach()
        loss_dict["explicit_paired_count"] = explicit_paired_count.detach()
        loss_dict["explicit_orphan_con_count"] = explicit_orphan_con_count.detach()
        loss_dict["explicit_orphan_seg_count"] = explicit_orphan_seg_count.detach()
        loss_dict["explicit_invalid_row_count"] = explicit_invalid_row_count.detach()
        loss_dict["explicit_pair_rate"] = explicit_pair_rate.detach()
        loss_dict["explicit_hidden_cosine"] = explicit_hidden_cosine.detach()
        loss_dict["latent_con_count"] = latent_con_count.detach()
        loss_dict["sam_prompt_encoder_calls"] = sam_prompt_encoder_calls.detach()
        loss_dict["sam_mask_decoder_calls"] = sam_mask_decoder_calls.detach()
        return loss_dict

    #
    # <---- HERE is the crucial fix: define get_visual_embs INSIDE the class
    #
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """
        Moved inside LISATForCausalLM so that self.get_visual_embs is valid.
        """
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        sam_mask_shape_list,
        max_new_tokens=32,
    ):
        with torch.no_grad():
            attention_mask = (input_ids != self.config.pad_token_id).long().to(input_ids.device)

            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                return_dict_in_generate=True,
                do_sample=False,
                temperature=0.2,
                use_cache=True,
            )
            output_ids = outputs.sequences

            # Re-run the full generated sequence to get hidden states aligned with
            # all generated [CON]/[SEG] tokens, not just the last generated step.
            if self.config.pad_token_id is not None:
                output_attention_mask = output_ids.ne(self.config.pad_token_id).long()
            else:
                output_attention_mask = torch.ones_like(output_ids).long()

            hidden_state_output = super().forward(
                images=images_clip,
                input_ids=output_ids,
                attention_mask=output_attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            output_hidden_states = hidden_state_output.hidden_states
            last_hidden_state = (
                output_hidden_states[-1]
                if isinstance(output_hidden_states, (list, tuple))
                else output_hidden_states
            )
            hidden_len = last_hidden_state.shape[1]

            try:
                if self.use_dia:
                    seg_token_mask, con_token_mask = self.build_dia_token_masks(
                        input_ids=output_ids,
                        hidden_len=hidden_len,
                    )
                else:
                    seg_token_mask = self.build_shifted_token_mask(
                        input_ids=output_ids,
                        token_idx=self.seg_token_idx,
                        hidden_len=hidden_len,
                    )
                    con_token_mask = None
            except RuntimeError:
                if not (self.use_dia and self.explicit_con_in_conversation):
                    raise
                output_pred_masks = []
                object_presence = [False for _ in sam_mask_shape_list]
                for _, original_size in sam_mask_shape_list:
                    original_size = (int(original_size[0]), int(original_size[1]))
                    output_pred_masks.append(
                        torch.zeros(
                            original_size,
                            dtype=torch.int,
                            device=images.device,
                        )
                    )
                return output_ids, output_pred_masks, object_presence

            seg_token_counts = seg_token_mask.int().sum(-1)
            object_presence = [count.item() > 0 for count in seg_token_counts]

            # evaluate() receives one generated conversation per image/question,
            # so each row is its own image-level group.
            offset = torch.arange(
                output_ids.shape[0] + 1,
                dtype=torch.long,
                device=output_ids.device,
            )

            seg_hidden = last_hidden_state[seg_token_mask]
            seg_flat = self.model.text_hidden_fcs[0](seg_hidden)
            seg_embeddings = self.split_embeddings_by_offset(seg_flat, seg_token_mask, offset)
            anchor_embeddings = None
            if self.use_dia:
                con_hidden = last_hidden_state[con_token_mask]
                if con_hidden.shape[0] != seg_hidden.shape[0]:
                    if self.explicit_con_in_conversation:
                        output_pred_masks = []
                        object_presence = [False for _ in sam_mask_shape_list]
                        for _, original_size in sam_mask_shape_list:
                            original_size = (int(original_size[0]), int(original_size[1]))
                            output_pred_masks.append(
                                torch.zeros(
                                    original_size,
                                    dtype=torch.int,
                                    device=images.device,
                                )
                            )
                        return output_ids, output_pred_masks, object_presence
                    raise RuntimeError(
                        "DIA evaluate hidden counts differ: "
                        f"con={con_hidden.shape[0]}, seg={seg_hidden.shape[0]}."
                    )
                con_flat = self.model.con_hidden_fcs[0](con_hidden)
                con_embeddings = self.split_embeddings_by_offset(
                    con_flat,
                    con_token_mask,
                    offset,
                )
                if self.dia_fusion_mode in {
                    "sparse_dense",
                    "bounded_sparse_dense",
                    "evidence_feedback",
                }:
                    anchor_flat = self.model.text_hidden_fcs[0](con_hidden)
                    anchor_embeddings = self.split_embeddings_by_offset(
                        anchor_flat,
                        con_token_mask,
                        offset,
                    )
            else:
                con_embeddings = None


            image_embeddings = self.get_visual_embs(images)

            pred_masks, _, _ = self.generate_pred_masks(
                seg_embeddings=seg_embeddings,
                con_embeddings=con_embeddings,
                image_embeddings=image_embeddings,
                sam_mask_shape_list=sam_mask_shape_list,
                anchor_embeddings=anchor_embeddings,
            )

            output_pred_masks = []
            for i, pred_mask in enumerate(pred_masks):
                original_size = sam_mask_shape_list[i][1]
                original_size = (int(original_size[0]), int(original_size[1]))

                if not object_presence[i] or pred_mask.shape[0] == 0:
                    output_pred_masks.append(
                        torch.zeros(
                            original_size,
                            dtype=torch.int,
                            device=images.device,
                        )
                    )
                    object_presence[i] = False
                    continue

                mask_i = (pred_mask[0] > 0).int()
                if mask_i.sum() == 0:
                    object_presence[i] = False

                output_pred_masks.append(mask_i)

        return output_ids, output_pred_masks, object_presence



def load_pretrained_model_LISAT(model_path, device_map="auto", device="cuda", **kwargs):
    kwargs["device_map"] = device_map
    config = LlavaConfig.from_pretrained(model_path)
    use_dia = kwargs.pop("use_dia", getattr(config, "use_dia", False))
    explicit_con_in_conversation = kwargs.pop(
        "explicit_con_in_conversation",
        getattr(config, "explicit_con_in_conversation", False),
    )
    dia_fusion_mode = kwargs.pop(
        "dia_fusion_mode",
        getattr(config, "dia_fusion_mode", "legacy"),
    )
    if explicit_con_in_conversation and not use_dia:
        raise RuntimeError("explicit_con_in_conversation requires use_dia=True.")
    if (
        dia_fusion_mode in {
            "sparse_dense",
            "bounded_sparse_dense",
            "faithful_evidence_fusion",
            "decoupled_evidence_prompt",
            "evidence_feedback",
        }
        and not explicit_con_in_conversation
    ):
        raise RuntimeError(
            f"{dia_fusion_mode} DIA requires explicit_con_in_conversation=True."
        )
    config.use_dia = use_dia
    config.explicit_con_in_conversation = explicit_con_in_conversation
    config.dia_fusion_mode = dia_fusion_mode
    config.faithful_fusion_hidden_dim = kwargs.get(
        "faithful_fusion_hidden_dim",
        getattr(config, "faithful_fusion_hidden_dim", 256),
    )
    config.faithful_max_delta_ratio = kwargs.get(
        "faithful_max_delta_ratio",
        getattr(config, "faithful_max_delta_ratio", 0.15),
    )
    config.faithful_delta_gain = kwargs.get(
        "faithful_delta_gain",
        getattr(config, "faithful_delta_gain", 1.0),
    )
    config.faithful_strict_config = kwargs.get(
        "faithful_strict_config",
        getattr(config, "faithful_strict_config", False),
    )
    config.latent_sparse_hidden_dim = kwargs.get(
        "latent_sparse_hidden_dim",
        getattr(config, "latent_sparse_hidden_dim", 256),
    )
    config.latent_sparse_max_delta_ratio = kwargs.get(
        "latent_sparse_max_delta_ratio",
        getattr(config, "latent_sparse_max_delta_ratio", 0.40),
    )
    config.latent_sparse_delta_gain = kwargs.get(
        "latent_sparse_delta_gain",
        getattr(config, "latent_sparse_delta_gain", 3.0),
    )
    config.latent_sparse_init_std = kwargs.get(
        "latent_sparse_init_std",
        getattr(config, "latent_sparse_init_std", 1e-3),
    )
    config.latent_dense_max_delta_ratio = kwargs.get(
        "latent_dense_max_delta_ratio",
        getattr(config, "latent_dense_max_delta_ratio", 0.15),
    )
    config.latent_dense_init_std = kwargs.get(
        "latent_dense_init_std",
        getattr(config, "latent_dense_init_std", 1e-3),
    )
    config.evidence_usage_loss_weight = kwargs.get(
        "evidence_usage_loss_weight",
        getattr(config, "evidence_usage_loss_weight", 0.10),
    )
    config.evidence_target_delta_ratio = kwargs.get(
        "evidence_target_delta_ratio",
        getattr(config, "evidence_target_delta_ratio", 0.12),
    )
    config.area_recall_loss_weight = kwargs.get(
        "area_recall_loss_weight",
        getattr(config, "area_recall_loss_weight", 0.20),
    )
    config.map_loss_weight = kwargs.get(
        "map_loss_weight",
        getattr(config, "map_loss_weight", 0.10),
    )
    config.anchor_loss_weight = kwargs.get(
        "anchor_loss_weight",
        getattr(config, "anchor_loss_weight", 0.10),
    )
    config.anchor_decay_steps = kwargs.get(
        "anchor_decay_steps",
        getattr(config, "anchor_decay_steps", 8000),
    )
    config.dia_loc_bias_init = kwargs.get(
        "dia_loc_bias_init",
        getattr(config, "dia_loc_bias_init", -4.0),
    )
    config.dia_fusion_max_strength = kwargs.get(
        "dia_fusion_max_strength",
        getattr(config, "dia_fusion_max_strength", 0.15),
    )
    config.dia_fusion_warmup_steps = kwargs.get(
        "dia_fusion_warmup_steps",
        getattr(config, "dia_fusion_warmup_steps", 2000),
    )
    config.dia_fusion_ramp_steps = kwargs.get(
        "dia_fusion_ramp_steps",
        getattr(config, "dia_fusion_ramp_steps", 4000),
    )
    config.dia_gate_floor = kwargs.get(
        "dia_gate_floor",
        getattr(config, "dia_gate_floor", 0.10),
    )
    config.dia_init_gate = kwargs.get(
        "dia_init_gate",
        getattr(config, "dia_init_gate", 0.50),
    )
    config.dia_training_stage = kwargs.get(
        "dia_training_stage",
        getattr(config, "dia_training_stage", "one_stage"),
    )
    config.dia_stage1_ce_loss_weight = kwargs.get(
        "dia_stage1_ce_loss_weight",
        getattr(config, "dia_stage1_ce_loss_weight", 0.25),
    )
    config.dia_stage1_attn_loss_weight = kwargs.get(
        "dia_stage1_attn_loss_weight",
        getattr(config, "dia_stage1_attn_loss_weight", 0.25),
    )
    config.dia_stage1_evidence_usage_loss_weight = kwargs.get(
        "dia_stage1_evidence_usage_loss_weight",
        getattr(config, "dia_stage1_evidence_usage_loss_weight", 0.10),
    )
    config.dia_stage1_area_recall_loss_weight = kwargs.get(
        "dia_stage1_area_recall_loss_weight",
        getattr(config, "dia_stage1_area_recall_loss_weight", 0.0),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    lisat_tokens = ["[SEG]"]
    if use_dia:
        lisat_tokens.append("[CON]")
    tokenizer.add_tokens(lisat_tokens)

    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    con_token_idx = (
        tokenizer("[CON]", add_special_tokens=False).input_ids[0]
        if use_dia
        else None
    )

    model = LISATForCausalLM.from_pretrained(
        model_path, 
        config=config,
        low_cpu_mem_usage=True, 
        seg_token_idx=seg_token_idx,
        con_token_idx=con_token_idx,
        use_dia=use_dia,
        explicit_con_in_conversation=explicit_con_in_conversation,
        dia_fusion_mode=dia_fusion_mode,
        **kwargs
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    if "training" in kwargs and kwargs["training"] is True:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=model.dtype)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, vision_tower, context_len


def _copy_con_from_seg_initialization(model, seg_token_idx, con_token_idx):
    if con_token_idx == seg_token_idx:
        raise RuntimeError("[CON] and [SEG] resolved to the same token id.")

    input_embeddings = model.get_input_embeddings()
    input_weight = input_embeddings.weight
    with torch.no_grad():
        input_weight[con_token_idx].copy_(input_weight[seg_token_idx])

        output_layer = model.get_output_embeddings()
        if (
            output_layer is not None
            and output_layer.weight.data_ptr() != input_weight.data_ptr()
        ):
            output_layer.weight[con_token_idx].copy_(
                output_layer.weight[seg_token_idx]
            )

    model.get_model().con_hidden_fcs.load_state_dict(
        model.get_model().text_hidden_fcs.state_dict(),
        strict=True,
    )

    assert torch.allclose(input_weight[con_token_idx], input_weight[seg_token_idx])
    for con_param, seg_param in zip(
        model.get_model().con_hidden_fcs.parameters(),
        model.get_model().text_hidden_fcs.parameters(),
    ):
        assert torch.equal(con_param, seg_param)


def _validate_dia_structure(model):
    base_model = model.get_model()
    expected_k = getattr(base_model.config, "dia_num_evidence_tokens", 1)
    expected_dropout = getattr(base_model.config, "dia_attn_dropout", 0.0)

    fusion_mode = getattr(base_model.config, "dia_fusion_mode", "legacy")
    if fusion_mode == "evidence_feedback":
        assert isinstance(base_model.context_adapter, SharedEvidenceAdapter)
        assert isinstance(base_model.evidence_fusion, EvidenceGuideFusionV2)
        assert base_model.context_adapter.num_evidence_tokens == 1
        assert expected_k == 1
        assert abs(expected_dropout) < 1e-12
        expected_bias = getattr(base_model.config, "dia_loc_bias_init", -4.0)
        actual_bias = base_model.context_adapter.loc_bias.detach().float().item()
        if abs(actual_bias - expected_bias) > 1e-4:
            raise RuntimeError(
                "SharedEvidenceAdapter loc_bias mismatch: "
                f"expected={expected_bias}, actual={actual_bias}."
            )
        fusion = base_model.evidence_fusion
        assert abs(
            fusion.max_strength
            - getattr(base_model.config, "dia_fusion_max_strength", 0.15)
        ) < 1e-8
        assert fusion.warmup_steps == int(
            getattr(base_model.config, "dia_fusion_warmup_steps", 2000)
        )
        assert fusion.ramp_steps == int(
            getattr(base_model.config, "dia_fusion_ramp_steps", 4000)
        )
        assert not hasattr(fusion, "res_scale")
        return

    assert base_model.context_adapter.num_evidence_tokens == expected_k
    assert abs(base_model.context_adapter.cross_attn.dropout - expected_dropout) < 1e-12

    if fusion_mode == "legacy":
        assert base_model.evidence_fusion.res_scale.detach().item() == 0.0
    elif fusion_mode == "decoupled_evidence_prompt":
        fusion = base_model.decoupled_mask_prompt
        assert fusion.evidence_norm.elementwise_affine is False
        assert fusion.evidence_in.bias is None
        assert fusion.evidence_out.bias is None
        assert torch.count_nonzero(fusion.evidence_out.weight.detach()).item() == 0
    elif fusion_mode == "faithful_evidence_fusion":
        fusion = base_model.faithful_evidence_fusion
        expected_hidden_dim = getattr(
            base_model.config,
            "faithful_fusion_hidden_dim",
            256,
        )
        expected_max_ratio = getattr(
            base_model.config,
            "faithful_max_delta_ratio",
            0.15,
        )
        expected_delta_gain = getattr(
            base_model.config,
            "faithful_delta_gain",
            1.0,
        )
        assert fusion.hidden_dim == expected_hidden_dim
        assert abs(fusion.max_delta_ratio - expected_max_ratio) < 1e-8
        assert abs(fusion.delta_gain - expected_delta_gain) < 1e-8
        assert torch.count_nonzero(fusion.fusion_out.weight.detach()).item() == 0
        assert fusion.evidence_norm.elementwise_affine is False
        assert fusion.evidence_proj.bias is None
        assert fusion.fusion_in.bias is None
        assert fusion.fusion_out.bias is None
    elif fusion_mode == "sparse_dense":
        bridge = base_model.explicit_token_bridge
        dense = base_model.dense_evidence_prompt
        expected_gate = getattr(base_model.config, "token_bridge_init_gate", 0.02)
        actual_gate = torch.sigmoid(
            bridge.gate[-1].bias.detach().float()
        ).mean().item()
        # The bridge bias can be materialized in fp16/bf16 during model
        # construction, so validate the intended small gate with dtype-aware
        # tolerance instead of requiring bit-exact sigmoid(logit(p)).
        if abs(actual_gate - expected_gate) > 1e-4:
            raise RuntimeError(
                "ExplicitTokenBridge initial gate mismatch: "
                f"expected={expected_gate}, actual={actual_gate}."
            )
        assert torch.count_nonzero(dense.out_proj.weight.detach()).item() == 0
        assert torch.count_nonzero(dense.out_proj.bias.detach()).item() == 0
    elif fusion_mode == "bounded_sparse_dense":
        role = base_model.explicit_role_adapter
        dense = base_model.bounded_dense_evidence_prompt
        expected_role_cap = getattr(base_model.config, "role_max_delta_ratio", 0.05)
        expected_dense_cap = getattr(base_model.config, "dense_max_delta_ratio", 0.10)
        expected_confidence_power = getattr(
            base_model.config,
            "dense_confidence_power",
            0.5,
        )
        assert torch.count_nonzero(role.role_out.weight.detach()).item() == 0
        assert torch.count_nonzero(role.role_out.bias.detach()).item() == 0
        assert torch.count_nonzero(dense.out_proj.weight.detach()).item() == 0
        assert abs(role.max_delta_ratio - expected_role_cap) < 1e-8
        assert abs(dense.max_delta_ratio - expected_dense_cap) < 1e-8
        assert abs(dense.confidence_power - expected_confidence_power) < 1e-8
    elif fusion_mode == "latent_sparse_dense_dia":
        sparse = base_model.latent_sparse_fusion
        dense = base_model.latent_dense_evidence_prompt
        expected_sparse_cap = getattr(
            base_model.config,
            "latent_sparse_max_delta_ratio",
            0.40,
        )
        expected_sparse_gain = getattr(
            base_model.config,
            "latent_sparse_delta_gain",
            3.0,
        )
        expected_target_ratio = getattr(
            base_model.config,
            "evidence_target_delta_ratio",
            0.12,
        )
        expected_dense_cap = getattr(
            base_model.config,
            "latent_dense_max_delta_ratio",
            0.15,
        )
        assert abs(sparse.max_delta_ratio - expected_sparse_cap) < 1e-8
        assert abs(sparse.delta_gain - expected_sparse_gain) < 1e-8
        assert abs(sparse.target_delta_ratio - expected_target_ratio) < 1e-8
        assert abs(dense.max_delta_ratio - expected_dense_cap) < 1e-8
        assert sparse.evidence_norm.elementwise_affine is False
        assert sparse.evidence_proj.bias is None
        assert sparse.fusion_in.bias is None
        assert sparse.fusion_out.bias is None
        assert dense.out_proj.bias is None
        if getattr(base_model.config, "visual_bottleneck_enabled", False):
            bottleneck = base_model.evidence_visual_bottleneck
            assert abs(
                bottleneck.beta
                - getattr(base_model.config, "visual_bottleneck_beta", 0.30)
            ) < 1e-8
            assert abs(
                bottleneck.max_delta_ratio
                - getattr(
                    base_model.config,
                    "visual_bottleneck_max_delta_ratio",
                    0.20,
                )
            ) < 1e-8
            assert abs(
                bottleneck.confidence_power
                - getattr(
                    base_model.config,
                    "visual_bottleneck_confidence_power",
                    0.25,
                )
            ) < 1e-8
            assert bottleneck.out_proj.bias is None
    else:
        raise AssertionError(f"Unknown fusion mode: {fusion_mode}")


def _validate_explicit_tokenizer_pair(tokenizer):
    con_ids = tokenizer("[CON]", add_special_tokens=False).input_ids
    seg_ids = tokenizer("[SEG]", add_special_tokens=False).input_ids
    pair_ids = tokenizer("[CON][SEG]", add_special_tokens=False).input_ids
    if len(con_ids) != 1 or len(seg_ids) != 1:
        raise RuntimeError(
            f"[CON]/[SEG] must be single tokens, got con={con_ids}, seg={seg_ids}."
        )
    if pair_ids != [con_ids[0], seg_ids[0]]:
        raise RuntimeError(
            f"[CON][SEG] must tokenize as adjacent pair, got {pair_ids}."
        )
    return con_ids[0], seg_ids[0], pair_ids


def init_LISAT_model(args, model_args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        use_fast=False,
        legacy=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer_size_before = len(tokenizer)
    if getattr(args, "explicit_con_in_conversation", False) and not args.use_dia:
        raise RuntimeError("explicit_con_in_conversation requires use_dia=True.")
    lisat_tokens = ["[SEG]"]
    if args.use_dia:
        lisat_tokens.append("[CON]")
    tokenizer.add_tokens(lisat_tokens)
    tokenizer_size_after_lisat_tokens = len(tokenizer)
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.con_token_idx = (
        tokenizer("[CON]", add_special_tokens=False).input_ids[0]
        if args.use_dia
        else None
    )
    model_args["seg_token_idx"] = args.seg_token_idx
    model_args["con_token_idx"] = args.con_token_idx
    model_args["use_dia"] = args.use_dia
    model_args["explicit_con_in_conversation"] = getattr(
        args,
        "explicit_con_in_conversation",
        False,
    )
    model_args["dia_fusion_mode"] = getattr(args, "dia_fusion_mode", "legacy")
    model_args["token_bridge_init_gate"] = getattr(
        args,
        "token_bridge_init_gate",
        0.02,
    )
    model_args["dense_attn_clip"] = getattr(args, "dense_attn_clip", 8.0)
    model_args["role_adapter_hidden_dim"] = getattr(
        args,
        "role_adapter_hidden_dim",
        256,
    )
    model_args["role_max_delta_ratio"] = getattr(args, "role_max_delta_ratio", 0.05)
    model_args["dense_max_delta_ratio"] = getattr(args, "dense_max_delta_ratio", 0.10)
    model_args["dense_confidence_power"] = getattr(
        args,
        "dense_confidence_power",
        0.5,
    )
    model_args["faithful_fusion_hidden_dim"] = getattr(
        args,
        "faithful_fusion_hidden_dim",
        256,
    )
    model_args["faithful_max_delta_ratio"] = getattr(
        args,
        "faithful_max_delta_ratio",
        0.15,
    )
    model_args["faithful_delta_gain"] = getattr(
        args,
        "faithful_delta_gain",
        1.0,
    )
    model_args["faithful_strict_config"] = getattr(
        args,
        "faithful_strict_config",
        False,
    )
    model_args["latent_sparse_hidden_dim"] = getattr(
        args,
        "latent_sparse_hidden_dim",
        256,
    )
    model_args["latent_sparse_max_delta_ratio"] = getattr(
        args,
        "latent_sparse_max_delta_ratio",
        0.40,
    )
    model_args["latent_sparse_delta_gain"] = getattr(
        args,
        "latent_sparse_delta_gain",
        3.0,
    )
    model_args["latent_sparse_init_std"] = getattr(
        args,
        "latent_sparse_init_std",
        1e-3,
    )
    model_args["latent_dense_max_delta_ratio"] = getattr(
        args,
        "latent_dense_max_delta_ratio",
        0.15,
    )
    model_args["latent_dense_init_std"] = getattr(
        args,
        "latent_dense_init_std",
        1e-3,
    )
    model_args["visual_bottleneck_enabled"] = getattr(
        args,
        "visual_bottleneck_enabled",
        False,
    )
    model_args["visual_bottleneck_beta"] = getattr(
        args,
        "visual_bottleneck_beta",
        0.30,
    )
    model_args["visual_bottleneck_attn_clip"] = getattr(
        args,
        "visual_bottleneck_attn_clip",
        8.0,
    )
    model_args["visual_bottleneck_max_delta_ratio"] = getattr(
        args,
        "visual_bottleneck_max_delta_ratio",
        0.20,
    )
    model_args["visual_bottleneck_confidence_power"] = getattr(
        args,
        "visual_bottleneck_confidence_power",
        0.25,
    )
    model_args["visual_bottleneck_init_std"] = getattr(
        args,
        "visual_bottleneck_init_std",
        1e-3,
    )
    model_args["evidence_usage_loss_weight"] = getattr(
        args,
        "evidence_usage_loss_weight",
        0.10,
    )
    model_args["evidence_target_delta_ratio"] = getattr(
        args,
        "evidence_target_delta_ratio",
        0.12,
    )
    model_args["area_recall_loss_weight"] = getattr(
        args,
        "area_recall_loss_weight",
        0.20,
    )
    model_args["map_loss_weight"] = getattr(args, "map_loss_weight", 0.10)
    model_args["anchor_loss_weight"] = getattr(args, "anchor_loss_weight", 0.10)
    model_args["anchor_decay_steps"] = getattr(args, "anchor_decay_steps", 8000)
    model_args["dia_loc_bias_init"] = getattr(args, "dia_loc_bias_init", -4.0)
    model_args["dia_fusion_max_strength"] = getattr(
        args,
        "dia_fusion_max_strength",
        0.15,
    )
    model_args["dia_fusion_warmup_steps"] = getattr(
        args,
        "dia_fusion_warmup_steps",
        2000,
    )
    model_args["dia_fusion_ramp_steps"] = getattr(
        args,
        "dia_fusion_ramp_steps",
        4000,
    )
    model_args["dia_gate_floor"] = getattr(args, "dia_gate_floor", 0.10)
    model_args["dia_init_gate"] = getattr(args, "dia_init_gate", 0.50)
    model_args["dia_training_stage"] = getattr(
        args,
        "dia_training_stage",
        "one_stage",
    )
    model_args["dia_stage1_ce_loss_weight"] = getattr(
        args,
        "dia_stage1_ce_loss_weight",
        0.25,
    )
    model_args["dia_stage1_attn_loss_weight"] = getattr(
        args,
        "dia_stage1_attn_loss_weight",
        0.25,
    )
    model_args["dia_stage1_evidence_usage_loss_weight"] = getattr(
        args,
        "dia_stage1_evidence_usage_loss_weight",
        0.10,
    )
    model_args["dia_stage1_area_recall_loss_weight"] = getattr(
        args,
        "dia_stage1_area_recall_loss_weight",
        0.0,
    )

    explicit_pair_ids = None
    if getattr(args, "explicit_con_in_conversation", False):
        _, _, explicit_pair_ids = _validate_explicit_tokenizer_pair(tokenizer)

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    tokenizer_size_after = len(tokenizer)

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    # The local LISAt/LLaVA checkpoints use model_type="llava_llama", which
    # is not registered in vanilla Transformers AutoConfig. Load it with the
    # repository's LLaVA config class, then inject DIA settings before model init.
    config = LlavaConfig.from_pretrained(args.version)
    config.train_mask_decoder = args.train_mask_decoder
    config.out_dim = args.out_dim
    config.mm_use_im_start_end = args.use_mm_start_end
    config.mm_vision_tower = args.vision_tower
    config.vision_tower = args.vision_tower
    config.use_dia = args.use_dia
    config.explicit_con_in_conversation = getattr(
        args,
        "explicit_con_in_conversation",
        False,
    )
    config.dia_fusion_mode = getattr(args, "dia_fusion_mode", "legacy")
    config.token_bridge_init_gate = getattr(args, "token_bridge_init_gate", 0.02)
    config.dense_attn_clip = getattr(args, "dense_attn_clip", 8.0)
    config.role_adapter_hidden_dim = getattr(args, "role_adapter_hidden_dim", 256)
    config.role_max_delta_ratio = getattr(args, "role_max_delta_ratio", 0.05)
    config.dense_max_delta_ratio = getattr(args, "dense_max_delta_ratio", 0.10)
    config.dense_confidence_power = getattr(args, "dense_confidence_power", 0.5)
    config.faithful_fusion_hidden_dim = getattr(
        args,
        "faithful_fusion_hidden_dim",
        256,
    )
    config.faithful_max_delta_ratio = getattr(
        args,
        "faithful_max_delta_ratio",
        0.15,
    )
    config.faithful_delta_gain = getattr(args, "faithful_delta_gain", 1.0)
    config.faithful_strict_config = getattr(args, "faithful_strict_config", False)
    config.latent_sparse_hidden_dim = getattr(args, "latent_sparse_hidden_dim", 256)
    config.latent_sparse_max_delta_ratio = getattr(
        args,
        "latent_sparse_max_delta_ratio",
        0.40,
    )
    config.latent_sparse_delta_gain = getattr(
        args,
        "latent_sparse_delta_gain",
        3.0,
    )
    config.latent_sparse_init_std = getattr(args, "latent_sparse_init_std", 1e-3)
    config.latent_dense_max_delta_ratio = getattr(
        args,
        "latent_dense_max_delta_ratio",
        0.15,
    )
    config.latent_dense_init_std = getattr(args, "latent_dense_init_std", 1e-3)
    config.visual_bottleneck_enabled = getattr(
        args,
        "visual_bottleneck_enabled",
        False,
    )
    config.visual_bottleneck_beta = getattr(args, "visual_bottleneck_beta", 0.30)
    config.visual_bottleneck_attn_clip = getattr(
        args,
        "visual_bottleneck_attn_clip",
        8.0,
    )
    config.visual_bottleneck_max_delta_ratio = getattr(
        args,
        "visual_bottleneck_max_delta_ratio",
        0.20,
    )
    config.visual_bottleneck_confidence_power = getattr(
        args,
        "visual_bottleneck_confidence_power",
        0.25,
    )
    config.visual_bottleneck_init_std = getattr(
        args,
        "visual_bottleneck_init_std",
        1e-3,
    )
    config.evidence_usage_loss_weight = getattr(
        args,
        "evidence_usage_loss_weight",
        0.10,
    )
    config.evidence_target_delta_ratio = getattr(
        args,
        "evidence_target_delta_ratio",
        0.12,
    )
    config.area_recall_loss_weight = getattr(args, "area_recall_loss_weight", 0.20)
    config.map_loss_weight = getattr(args, "map_loss_weight", 0.10)
    config.anchor_loss_weight = getattr(args, "anchor_loss_weight", 0.10)
    config.anchor_decay_steps = getattr(args, "anchor_decay_steps", 8000)
    config.dia_loc_bias_init = getattr(args, "dia_loc_bias_init", -4.0)
    config.dia_fusion_max_strength = getattr(
        args,
        "dia_fusion_max_strength",
        0.15,
    )
    config.dia_fusion_warmup_steps = getattr(
        args,
        "dia_fusion_warmup_steps",
        2000,
    )
    config.dia_fusion_ramp_steps = getattr(args, "dia_fusion_ramp_steps", 4000)
    config.dia_gate_floor = getattr(args, "dia_gate_floor", 0.10)
    config.dia_init_gate = getattr(args, "dia_init_gate", 0.50)
    config.dia_training_stage = getattr(args, "dia_training_stage", "one_stage")
    config.dia_stage1_ce_loss_weight = getattr(
        args,
        "dia_stage1_ce_loss_weight",
        0.25,
    )
    config.dia_stage1_attn_loss_weight = getattr(
        args,
        "dia_stage1_attn_loss_weight",
        0.25,
    )
    config.dia_stage1_evidence_usage_loss_weight = getattr(
        args,
        "dia_stage1_evidence_usage_loss_weight",
        0.10,
    )
    config.dia_stage1_area_recall_loss_weight = getattr(
        args,
        "dia_stage1_area_recall_loss_weight",
        0.0,
    )
    config.dia_num_evidence_tokens = args.dia_num_evidence_tokens
    config.dia_num_heads = args.dia_num_heads
    config.dia_attn_dropout = args.dia_attn_dropout
    config.fusion_dropout = args.fusion_dropout
    config.attn_loss_weight = args.attn_loss_weight if args.use_dia else 0.0
    config.dia_bypass_fusion = getattr(args, "dia_bypass_fusion", False)

    previous_transformers_verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        pretrained_output = LISATForCausalLM.from_pretrained(
            args.version,
            config=config,
            torch_dtype=torch_dtype,
            # DIA adds a scalar ReZero parameter. Transformers' meta-device loader
            # can fail on 0-d parameters, so use the regular loader here.
            low_cpu_mem_usage=False,
            output_loading_info=True,
            **model_args
        )
    finally:
        transformers.logging.set_verbosity(previous_transformers_verbosity)
    if isinstance(pretrained_output, tuple):
        model, loading_info = pretrained_output
    else:
        model = pretrained_output
        loading_info = {}
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if "lisat_pre" in args.version.lower():
        if getattr(args, "local_rank", 0) == 0:
            print(
                "[LISAT_PRE] Loaded base checkpoint without post-load "
                "initialize_lisat_modules(); stage-2 heads are the constructor "
                "instances and will not be overwritten after from_pretrained."
            )

    model.resize_token_embeddings(len(tokenizer))
    if args.use_dia and getattr(args, "init_con_from_seg", True):
        _copy_con_from_seg_initialization(
            model,
            args.seg_token_idx,
            args.con_token_idx,
        )
    if args.use_dia:
        _validate_dia_structure(model)
    if getattr(args, "local_rank", 0) == 0:
        missing_keys = loading_info.get("missing_keys", [])
        unexpected_keys = loading_info.get("unexpected_keys", [])
        print(
            "[Tokenizer] "
            f"size_before={tokenizer_size_before}, "
            f"size_after_lisat_tokens={tokenizer_size_after_lisat_tokens}, "
            f"size_after={tokenizer_size_after}, "
            f"seg_token_idx={args.seg_token_idx}, "
            f"con_token_idx={args.con_token_idx}"
        )
        if explicit_pair_ids is not None:
            print(
                "[Tokenizer] "
                f"explicit_pair=[CON][SEG], token_ids={explicit_pair_ids}"
            )
        print(
            "[LoadInfo] "
            f"missing_keys={len(missing_keys)}, "
            f"unexpected_keys={len(unexpected_keys)}"
        )

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Configure LoRA if applicable
    if args.lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            exclude_list = ["visual_model", "vision_tower", "mm_projector", "text_hidden_fcs"]
            if args.use_dia:
                exclude_list.extend(
                    [
                        "con_hidden_fcs",
                        "context_adapter",
                        "evidence_fusion",
                        "explicit_token_bridge",
                        "dense_evidence_prompt",
                        "explicit_role_adapter",
                        "bounded_dense_evidence_prompt",
                        "faithful_evidence_fusion",
                        "latent_sparse_fusion",
                        "latent_dense_evidence_prompt",
                        "evidence_visual_bottleneck",
                    ]
                )
            for name, module in model.named_modules():
                if isinstance(module, cls) and not any(x in name for x in exclude_list) \
                   and any([x in name for x in lora_target_modules]):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    dia_parts = (
        "con_hidden_fcs",
        "context_adapter",
        "evidence_fusion",
        "explicit_token_bridge",
        "dense_evidence_prompt",
        "explicit_role_adapter",
        "bounded_dense_evidence_prompt",
        "faithful_evidence_fusion",
        "latent_sparse_fusion",
        "latent_dense_evidence_prompt",
        "evidence_visual_bottleneck",
    )
    managed_trainable_parts = (
        "lm_head",
        "embed_tokens",
        "mask_decoder",
        "text_hidden_fcs",
        *dia_parts,
    )

    # Stage 1 is an evidence-only warmup: [CON] and attention learn to find
    # visual evidence while the original [SEG] decoding path stays fixed.
    stage = getattr(args, "dia_training_stage", "one_stage")
    fusion_mode = getattr(args, "dia_fusion_mode", "legacy")
    if (
        args.use_dia
        and fusion_mode == "latent_sparse_dense_dia"
        and stage == "evidence"
    ):
        trainable_parts = [
            "lm_head",
            "embed_tokens",
            "con_hidden_fcs",
            "context_adapter",
        ]
        if getattr(args, "visual_bottleneck_enabled", False):
            trainable_parts.append("evidence_visual_bottleneck")
    else:
        trainable_parts = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]

    if args.use_dia:
        trainable_parts.extend(["con_hidden_fcs", "context_adapter"])
        if not (
            fusion_mode == "latent_sparse_dense_dia"
            and stage == "evidence"
        ):
            if fusion_mode == "legacy":
                trainable_parts.append("evidence_fusion")
            elif fusion_mode == "evidence_feedback":
                trainable_parts.append("evidence_fusion")
            elif fusion_mode == "decoupled_evidence_prompt":
                trainable_parts.append("decoupled_mask_prompt")
            elif fusion_mode == "faithful_evidence_fusion":
                trainable_parts.append("faithful_evidence_fusion")
            elif fusion_mode == "sparse_dense":
                trainable_parts.extend(["explicit_token_bridge", "dense_evidence_prompt"])
            elif fusion_mode == "bounded_sparse_dense":
                trainable_parts.extend(
                    ["explicit_role_adapter", "bounded_dense_evidence_prompt"]
                )
            elif fusion_mode == "latent_sparse_dense_dia":
                trainable_parts.extend(
                    ["latent_sparse_fusion", "latent_dense_evidence_prompt"]
                )
                if getattr(args, "visual_bottleneck_enabled", False):
                    trainable_parts.append("evidence_visual_bottleneck")
            else:
                raise ValueError(f"Unsupported DIA fusion mode: {fusion_mode}")

    for n, p in model.named_parameters():
        if any(part in n for part in managed_trainable_parts):
            p.requires_grad = False

    for n, p in model.named_parameters():
        if any(part in n for part in trainable_parts):
            p.requires_grad = True

    if getattr(args, "local_rank", 0) == 0:
        core_trainable_params = 0
        dia_trainable_params = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(part in name for part in dia_parts):
                dia_trainable_params += param.numel()
            else:
                core_trainable_params += param.numel()
        print(
            "[Trainable] "
            f"dia_training_stage={stage}, "
            f"selected_parts={sorted(set(trainable_parts))}, "
            f"core_trainable_params={core_trainable_params}, "
            f"dia_trainable_params={dia_trainable_params}"
        )

    return tokenizer, model, vision_tower
