from json import decoder
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

from .DIA_LISAt import ContextEvidenceAdapter, EvidenceGuideFusion, attention_alignment_loss

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
):
    mask_bce_sum = ce_loss.new_zeros(())
    mask_dice_sum = ce_loss.new_zeros(())
    attn_loss_sum = ce_loss.new_zeros(())
    num_masks = 0
    num_attn_masks = 0

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

        attn_maps = attn_maps_list[batch_idx]
        if attn_maps is not None:
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

    mask_loss = mask_bce_loss + mask_dice_loss
    total_loss = ce_loss + mask_loss + attn_loss_weight * attn_loss

    return {
        "loss": total_loss,
        "mask_bce_loss": mask_bce_loss,
        "mask_dice_loss": mask_dice_loss,
        "mask_loss": mask_loss,
        "attn_alignment_loss": attn_loss,
        "attn_loss": attn_loss,
        "num_positive_masks": ce_loss.new_tensor(float(num_masks)),
        "num_valid_attn_masks": ce_loss.new_tensor(float(num_attn_masks)),
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
            self.context_adapter = ContextEvidenceAdapter(
                dim=out_dim,
                num_heads=getattr(config, "dia_num_heads", 8),
                num_evidence_tokens=getattr(config, "dia_num_evidence_tokens", 1),
                dropout=getattr(config, "dia_attn_dropout", 0.0),
            )
            self.evidence_fusion = EvidenceGuideFusion(
                dim=out_dim,
                dropout=getattr(config, "fusion_dropout", 0.0),
            )
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        self.lisat_modules_initialized = True

    @property
    def evidence_adapter(self):
        if not self.use_dia:
            raise AttributeError("DIA evidence adapter is disabled because use_dia=False.")
        return self.context_adapter


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
        self.con_token_idx = kwargs.pop("con_token_idx", None)
        self.dia_bypass_fusion = kwargs.pop(
            "dia_bypass_fusion",
            getattr(config, "dia_bypass_fusion", False),
        )
        config.dia_bypass_fusion = self.dia_bypass_fusion
        self.attn_loss_weight = kwargs.pop(
            "attn_loss_weight",
            getattr(config, "attn_loss_weight", 0.02),
        )
        if not self.use_dia:
            self.attn_loss_weight = 0.0
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
                    self.model.evidence_fusion,
                ]
            )
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



    def build_paired_con_seg_prompt_mask(self, input_ids, hidden_len):
        """Locate the original LISAt prompt state after inserting [CON] [SEG].

        ?? LISAt ?????? [SEG] ? hidden state??DIA ??? token
        ? [SEG] ?? [CON] [SEG] ??????????? [CON] ? hidden
        state???? [SEG] projector ? [CON] projector ????????
        hidden state?????? mask prompt ? evidence query ????
        """
        if self.con_token_idx is None:
            return torch.zeros(
                input_ids.shape[0],
                hidden_len,
                dtype=torch.bool,
                device=input_ids.device,
            )

        con_next = input_ids[:, 1:] == self.con_token_idx
        seg_after_con = torch.zeros_like(con_next)
        # Tokenizers can encode "[CON] [SEG]" as [CON], whitespace, [SEG].
        # Treat a nearby following [SEG] as paired with [CON], so the prompt
        # state remains the one that predicts [CON].
        max_pair_gap = 4
        for gap in range(1, max_pair_gap + 1):
            if input_ids.shape[1] > 1 + gap:
                seg_after_con[:, :-gap] = seg_after_con[:, :-gap] | (
                    input_ids[:, 1 + gap:] == self.seg_token_idx
                )
        token_mask = con_next & seg_after_con

        left_pad = self.get_vision_tower().num_patches - 1
        token_mask = torch.cat(
            [
                torch.zeros(
                    token_mask.shape[0],
                    left_pad,
                    dtype=torch.bool,
                    device=token_mask.device,
                ),
                token_mask,
            ],
            dim=1,
        )

        if token_mask.shape[1] < hidden_len:
            right_pad = hidden_len - token_mask.shape[1]
            token_mask = F.pad(token_mask, (0, right_pad), value=False)

        return token_mask[:, :hidden_len]

    def build_dia_prompt_token_mask(self, input_ids, seg_token_mask, hidden_len):
        """Use paired [CON] [SEG] states, with old [SEG] states as fallback."""
        prompt_token_mask = self.build_paired_con_seg_prompt_mask(
            input_ids=input_ids,
            hidden_len=hidden_len,
        )
        missing_pair = prompt_token_mask.int().sum(-1) == 0
        if missing_pair.any():
            # ????????????? [SEG] ????????????????
            prompt_token_mask = prompt_token_mask.clone()
            prompt_token_mask[missing_pair] = seg_token_mask[missing_pair]
        return prompt_token_mask

    
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

    def generate_pred_masks(self, seg_embeddings, con_embeddings, image_embeddings, sam_mask_shape_list):
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
        }
        for i in range(len(seg_embeddings)):
            seg_i = seg_embeddings[i]
            con_i = con_embeddings[i] if self.use_dia and con_embeddings is not None else None

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
                # Reuse one concept query for multiple prompts from the same image.
                if con_i.shape[0] == 1 and seg_i.shape[0] > 1:
                    con_i = con_i.expand(seg_i.shape[0], -1)
                # Generated answers can still contain [SEG] without [CON]. Keep
                # inference robust, but do not inject [CON] into the dataset.
                if con_i.shape[0] == 0:
                    con_i = seg_i

                num_prompts = min(seg_i.shape[0], con_i.shape[0])
                seg_i = seg_i[:num_prompts]
                con_i = con_i[:num_prompts]
            else:
                num_prompts = seg_i.shape[0]
                seg_i = seg_i[:num_prompts]

            decoder_dtype = next(self.model.visual_model.mask_decoder.parameters()).dtype
            decoder_device = image_embeddings.device

            seg_i = seg_i.to(device=decoder_device, dtype=decoder_dtype)
            if self.use_dia:
                con_i = con_i.to(device=decoder_device, dtype=decoder_dtype)

            image_i = image_embeddings[i].unsqueeze(0).to(dtype=decoder_dtype)
            image_pe = self.model.visual_model.prompt_encoder.get_dense_pe().to(
                device=decoder_device, dtype=decoder_dtype
            )

            if self.use_dia:
                evidence_tokens, attn_maps = self.model.context_adapter(
                    con_embeddings=con_i,
                    image_embeddings=image_i,
                    image_pe=image_pe,
                )
                if getattr(self, "dia_bypass_fusion", False):
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
                flat_attn = attn_maps.detach().flatten(-2).clamp_min(1e-8)
                debug_stats["attention_entropies"].append(
                    -(flat_attn * flat_attn.log()).sum(dim=-1).mean()
                )
            else:
                prompt_tokens = seg_i.unsqueeze(1)
                attn_maps = None

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

            low_res_masks, _ = self.model.visual_model.mask_decoder(
                image_embeddings=image_i,
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

        seg_token_mask = self.build_shifted_token_mask(
            input_ids=input_ids,
            token_idx=self.seg_token_idx,
            hidden_len=hidden_len,
        )
        if self.use_dia:
            prompt_token_mask = self.build_dia_prompt_token_mask(
                input_ids=input_ids,
                seg_token_mask=seg_token_mask,
                hidden_len=hidden_len,
            )
        else:
            prompt_token_mask = seg_token_mask

        prompt_hidden = last_hidden_state
        seg_flat = self.model.text_hidden_fcs[0](prompt_hidden)[prompt_token_mask]
        seg_embeddings = self.split_embeddings_by_offset(seg_flat, prompt_token_mask, offset)
        if self.use_dia:
            # DIA uses the same prompt hidden state for both projectors in this
            # compatibility version: dia_con_source=seg_hidden_dual_projector.
            con_flat = self.model.con_hidden_fcs[0](prompt_hidden)[prompt_token_mask]
            con_embeddings = self.split_embeddings_by_offset(
                con_flat,
                prompt_token_mask,
                offset,
            )
        else:
            con_embeddings = None


        image_embeddings = self.get_visual_embs(images)

        # DIA-LISA mask decoding。
        pred_masks, attn_maps_list, dia_debug_stats = self.generate_pred_masks(
            seg_embeddings=seg_embeddings,
            con_embeddings=con_embeddings,
            image_embeddings=image_embeddings,
            sam_mask_shape_list=sam_mask_shape_list,
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
        )

        if not has_positive_masks:
            loss_dict["loss"] = loss_dict["loss"] + self.segmentation_zero_anchor(
                loss_dict["loss"]
            )
        elif self.use_dia and not uses_con_projector:
            loss_dict["loss"] = loss_dict["loss"] + self.con_projector_zero_anchor(
                loss_dict["loss"]
            )

        if dia_debug_stats["gate_means"]:
            gate_mean = torch.stack(dia_debug_stats["gate_means"]).mean()
        else:
            gate_mean = ce_loss.new_zeros(())
        if dia_debug_stats["attention_entropies"]:
            attention_entropy = torch.stack(dia_debug_stats["attention_entropies"]).mean()
        else:
            attention_entropy = ce_loss.new_zeros(())
        if self.use_dia:
            res_scale = torch.tanh(self.model.evidence_fusion.res_scale.detach()).to(ce_loss.device)
        else:
            res_scale = ce_loss.new_zeros(())

        loss_dict["ce_loss"] = ce_loss
        loss_dict["res_scale"] = res_scale
        loss_dict["gate_mean"] = gate_mean.detach()
        loss_dict["attention_entropy"] = attention_entropy.detach()
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

            seg_token_mask = self.build_shifted_token_mask(
                input_ids=output_ids,
                token_idx=self.seg_token_idx,
                hidden_len=hidden_len,
            )

            if self.use_dia:
                prompt_token_mask = self.build_dia_prompt_token_mask(
                    input_ids=output_ids,
                    seg_token_mask=seg_token_mask,
                    hidden_len=hidden_len,
                )
            else:
                prompt_token_mask = seg_token_mask

            seg_token_counts = seg_token_mask.int().sum(-1)
            object_presence = [count.item() > 0 for count in seg_token_counts]

            # evaluate() ????????????? prompt ???
            prompt_hidden = last_hidden_state
            seg_flat = self.model.text_hidden_fcs[0](prompt_hidden)[prompt_token_mask]

            # evaluate() receives one generated conversation per image/question,
            # so each row is its own image-level group.
            offset = torch.arange(
                output_ids.shape[0] + 1,
                dtype=torch.long,
                device=output_ids.device,
            )

            seg_embeddings = self.split_embeddings_by_offset(seg_flat, prompt_token_mask, offset)
            if self.use_dia:
                con_flat = self.model.con_hidden_fcs[0](prompt_hidden)[prompt_token_mask]
                con_embeddings = self.split_embeddings_by_offset(
                    con_flat,
                    prompt_token_mask,
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
    use_dia = kwargs.pop("use_dia", False)

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
        low_cpu_mem_usage=True, 
        seg_token_idx=seg_token_idx,
        con_token_idx=con_token_idx,
        use_dia=use_dia,
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
    assert base_model.evidence_adapter.num_evidence_tokens == 1
    assert base_model.evidence_adapter.cross_attn.dropout == 0.0
    assert base_model.evidence_fusion.res_scale.detach().item() == 0.0


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
    config.dia_num_evidence_tokens = args.dia_num_evidence_tokens
    config.dia_num_heads = args.dia_num_heads
    config.dia_attn_dropout = args.dia_attn_dropout
    config.fusion_dropout = args.fusion_dropout
    config.attn_loss_weight = args.attn_loss_weight if args.use_dia else 0.0
    config.dia_bypass_fusion = getattr(args, "dia_bypass_fusion", False)

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
                exclude_list.extend(["con_hidden_fcs", "context_adapter", "evidence_fusion"])
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

    # Make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    trainable_parts = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
    if args.use_dia:
        trainable_parts.extend(["con_hidden_fcs", "context_adapter", "evidence_fusion"])
    for n, p in model.named_parameters():
        if any(part in n for part in trainable_parts):
            p.requires_grad = True

    if getattr(args, "local_rank", 0) == 0:
        dia_parts = ("con_hidden_fcs", "context_adapter", "evidence_fusion")
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
            f"core_trainable_params={core_trainable_params}, "
            f"dia_trainable_params={dia_trainable_params}"
        )

    return tokenizer, model, vision_tower
