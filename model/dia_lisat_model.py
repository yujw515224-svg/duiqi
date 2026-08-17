# -*- coding: utf-8 -*-
"""DIA-LISAt: decoupled image-text alignment on top of LISAt.

The model keeps every LISAt component untouched (RemoteCLIP tower, Vicuna-7B +
LoRA, SAM encoder/decoder, ``text_hidden_fcs``) and only rewires the
*token-to-mask* path:

    h_CON --> ConceptToEvidenceAdapter(F) --> evidence e, attention map A
    h_SEG --> text_hidden_fcs            --> prompt p
    z = EvidenceGuidedFusion(p, e)       --> SAM prompt encoder + mask decoder

Losses:  L = L_txt + L_mask(bce, dice) + lambda_attn * L_attn(A, M_gt)

Because the fusion residual is zero-initialised, ``z == p`` at step 0 and the
model starts *exactly* from the LISAt baseline.
"""

from typing import Dict, List, Optional, Tuple

import torch
import transformers
from peft import LoraConfig, get_peft_model

from model.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
)

from .LISAT import LISATForCausalLM, dice_loss, sigmoid_ce_loss
from .dia_modules import (
    ConceptToEvidenceAdapter,
    EvidenceGuidedFusion,
    attention_alignment_loss,
    attention_mass_in_mask,
    build_special_token_mask,
    compute_dia_prompts,
    decode_masks_with_sam,
    rows_to_image_index,
    split_by_token_offset,
)

__all__ = [
    "DIA_DEFAULTS",
    "DIALISATForCausalLM",
    "init_dia_lisat_model",
    "load_pretrained_model_DIA_LISAT",
]


# Every DIA hyper-parameter, with its default. They are mirrored onto ``config``
# so that a checkpoint reloaded through ``from_pretrained`` rebuilds the very
# same modules without having to pass the flags again.
DIA_DEFAULTS: Dict[str, object] = {
    "dia_embed_dim": 256,        # width of the adapter / evidence token
    "dia_num_heads": 8,          # cross-attention heads
    "dia_dropout": 0.0,
    "dia_fusion_hidden_dim": 256,
    "dia_max_delta_ratio": 0.5,  # ||z - p|| <= ratio * ||p||   (<=0 disables)
    "dia_use_dense_pe": True,    # add SAM positional encoding to the K/V tokens
    "dia_attn_loss_weight": 0.1,
    "dia_attn_loss_mode": "mass",  # "mass" | "kl"
}


def _apply_dia_config(config, kwargs: dict) -> None:
    """Resolve DIA hyper-parameters: explicit kwarg > value stored on config > default."""
    for name, default in DIA_DEFAULTS.items():
        value = kwargs.pop(name, None)
        if value is None:
            value = getattr(config, name, default)
        setattr(config, name, value)


class DIALISATForCausalLM(LISATForCausalLM):
    """LISAt + concept token ``[CON]`` + concept-to-evidence adapter."""

    def __init__(self, config, **kwargs):
        con_token_idx = kwargs.pop("con_token_idx", None)
        _apply_dia_config(config, kwargs)
        super().__init__(config, **kwargs)

        if con_token_idx is None:
            con_token_idx = getattr(config, "con_token_idx", None)
        self.con_token_idx = con_token_idx
        config.con_token_idx = con_token_idx

        # LISATForCausalLM only sets the loss weights when the config comes from
        # a *pre-DIA* checkpoint; fill the gaps so resuming a DIA run still trains.
        for name, default in (
            ("ce_loss_weight", 1.0),
            ("dice_loss_weight", 0.5),
            ("bce_loss_weight", 2.0),
        ):
            if getattr(self, name, None) is None:
                setattr(self, name, float(kwargs.get(name, default)))

        self.attn_loss_weight = float(config.dia_attn_loss_weight)
        self.attn_loss_mode = str(config.dia_attn_loss_mode)
        self.initialize_dia_modules(config)

    # ------------------------------------------------------------------ #
    # module construction
    # ------------------------------------------------------------------ #
    def initialize_dia_modules(self, config) -> None:
        """Create the two DIA modules on ``self.model`` (idempotent)."""
        if getattr(self.model, "dia_adapter", None) is not None:
            return
        out_dim = getattr(config, "out_dim", 256)
        self.model.dia_adapter = ConceptToEvidenceAdapter(
            llm_dim=config.hidden_size,
            visual_dim=out_dim,
            embed_dim=int(config.dia_embed_dim),
            num_heads=int(config.dia_num_heads),
            dropout=float(config.dia_dropout),
            use_dense_pe=bool(config.dia_use_dense_pe),
        )
        self.model.dia_fusion = EvidenceGuidedFusion(
            prompt_dim=out_dim,
            evidence_dim=int(config.dia_embed_dim),
            hidden_dim=int(config.dia_fusion_hidden_dim),
            dropout=float(config.dia_dropout),
            max_delta_ratio=float(config.dia_max_delta_ratio),
        )
        for module in (self.model.dia_adapter, self.model.dia_fusion):
            module.train()
            for param in module.parameters():
                param.requires_grad = True

    def _init_weights(self, module):
        """Keep HuggingFace's missing-key re-initialisation away from DIA.

        ``PreTrainedModel`` re-initialises every module absent from the
        checkpoint with Llama's ``normal_(0, initializer_range)``, which would
        wipe out the zero-initialised fusion residual (and with it the
        "starts exactly at the LISAt baseline" guarantee). ``apply`` visits
        children before parents, so re-applying our init here always wins.
        """
        if isinstance(module, (ConceptToEvidenceAdapter, EvidenceGuidedFusion)):
            module.reset_dia_parameters()
            return
        super()._init_weights(module)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _last_hidden_state(hidden_states) -> torch.Tensor:
        """Normalise the many shapes ``hidden_states`` can take into a tensor."""
        if torch.is_tensor(hidden_states):
            return hidden_states
        last = hidden_states[-1]
        return last if torch.is_tensor(last) else last[-1]

    def _dense_pe(self) -> Optional[torch.Tensor]:
        if not getattr(self.config, "dia_use_dense_pe", True):
            return None
        return self.model.visual_model.prompt_encoder.get_dense_pe()

    def compute_dia_prompts(
        self,
        hidden_states: torch.Tensor,
        image_embeddings: torch.Tensor,
        seg_token_mask: torch.Tensor,
        con_token_mask: Optional[torch.Tensor],
        row_to_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Thin wrapper around :func:`model.dia_modules.compute_dia_prompts`."""
        return compute_dia_prompts(
            hidden_states=hidden_states,
            image_embeddings=image_embeddings,
            seg_token_mask=seg_token_mask,
            con_token_mask=con_token_mask,
            row_to_image=row_to_image,
            text_hidden_fc=self.model.text_hidden_fcs[0],
            adapter=self.model.dia_adapter,
            fusion=self.model.dia_fusion,
            image_pe=self._dense_pe(),
        )

    def decode_masks(
        self,
        prompt_embeddings: List[Optional[torch.Tensor]],
        image_embeddings: torch.Tensor,
        sam_mask_shape_list: List[tuple],
    ) -> List[torch.Tensor]:
        return decode_masks_with_sam(
            self.model.visual_model,
            prompt_embeddings,
            image_embeddings,
            sam_mask_shape_list,
        )

    # ------------------------------------------------------------------ #
    # training / offline-inference forward
    # ------------------------------------------------------------------ #
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

        num_tokens_per_image = self.get_vision_tower().num_patches
        seg_token_mask = build_special_token_mask(
            input_ids, self.seg_token_idx, num_tokens_per_image
        )
        con_token_mask = (
            build_special_token_mask(input_ids, self.con_token_idx, num_tokens_per_image)
            if self.con_token_idx is not None
            else None
        )

        # ---- LLM forward (identical to LISAt) ---------------------------- #
        # NOTE: skip LISATForCausalLM.forward, which would dispatch back here.
        llm_forward = super(LISATForCausalLM, self).forward
        if inference:
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            output = llm_forward(
                images=images_clip_extend,
                attention_mask=attention_masks,
                input_ids=input_ids,
                output_hidden_states=True,
            )
            torch.cuda.empty_cache()
            hidden_states = self._last_hidden_state(output.hidden_states)
            output = None
        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_list.append(
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
            images_clip = torch.cat(images_clip_list, dim=0)
            output = llm_forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            hidden_states = self._last_hidden_state(output.hidden_states)

        # ---- DIA: concept -> evidence -> fused prompt --------------------- #
        image_embeddings = self.get_visual_embs(images)
        row_to_image = rows_to_image_index(input_ids.shape[0], offset)
        prompts, attn_maps, dia_stats = self.compute_dia_prompts(
            hidden_states,
            image_embeddings,
            seg_token_mask,
            con_token_mask,
            row_to_image,
        )

        prompts_per_image = split_by_token_offset(prompts, seg_token_mask, offset)
        attn_per_image = split_by_token_offset(attn_maps, seg_token_mask, offset)
        pred_masks = self.decode_masks(
            prompts_per_image, image_embeddings, sam_mask_shape_list
        )

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": masks_list,
                "attn_maps": attn_per_image,
            }

        # ---- losses ------------------------------------------------------- #
        ce_loss = output.loss * self.ce_loss_weight
        mask_bce_loss = ce_loss.new_zeros(())
        mask_dice_loss = ce_loss.new_zeros(())
        attn_loss_sum = ce_loss.new_zeros(())
        attn_mass_sum = ce_loss.new_zeros(())
        num_masks = 0
        num_attn = 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = masks_list[batch_idx]
            pred_mask = pred_masks[batch_idx]
            if gt_mask.shape[0] == 0:
                continue
            assert gt_mask.shape[0] == pred_mask.shape[0], (
                f"Mismatch: gt_mask {gt_mask.shape}, pred_mask {pred_mask.shape}"
            )
            gt_mask = gt_mask.to(pred_mask.device)
            num_gt = gt_mask.shape[0]

            # mask losses in fp32: remote-sensing targets are tiny and bf16
            # rounding wipes out their gradient contribution.
            pred_f = pred_mask.float()
            gt_f = gt_mask.float()
            mask_bce_loss = mask_bce_loss + sigmoid_ce_loss(pred_f, gt_f, num_gt) * num_gt
            mask_dice_loss = mask_dice_loss + dice_loss(pred_f, gt_f, num_gt) * num_gt
            num_masks += num_gt

            attn = attn_per_image[batch_idx]
            if attn.shape[0] == num_gt:
                loss_i, valid_i = attention_alignment_loss(
                    attn, gt_mask, mode=self.attn_loss_mode
                )
                if valid_i > 0:
                    attn_loss_sum = attn_loss_sum + loss_i.to(attn_loss_sum.dtype) * valid_i
                    mass_i, _ = attention_mass_in_mask(attn, gt_mask)
                    attn_mass_sum = attn_mass_sum + mass_i.to(attn_mass_sum.dtype) * valid_i
                    num_attn += valid_i

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        attn_loss = attn_loss_sum / num_attn if num_attn > 0 else ce_loss.new_zeros(())
        attn_mass = attn_mass_sum / num_attn if num_attn > 0 else ce_loss.new_zeros(())
        total_loss = ce_loss + mask_loss + self.attn_loss_weight * attn_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "attn_loss": attn_loss,
            "attn_mass": attn_mass,
            "dia_gate": dia_stats["gate"],
            "dia_delta_ratio": dia_stats["delta_ratio"],
            "dia_con_hit_rate": dia_stats["con_hit_rate"],
            "num_valid_attn_masks": ce_loss.new_tensor(float(num_attn)),
        }

    # ------------------------------------------------------------------ #
    # generation-time evaluation (same signature as LISAt)
    # ------------------------------------------------------------------ #
    def _hidden_states_from_generate(
        self, outputs, expected_len: int
    ) -> Optional[torch.Tensor]:
        hidden = getattr(outputs, "hidden_states", None)
        if hidden is None:
            return None
        candidate = self._last_hidden_state(hidden)
        if candidate.shape[1] == expected_len:
            return candidate
        if not torch.is_tensor(hidden):
            parts = [
                step if torch.is_tensor(step) else step[-1] for step in hidden
            ]
            merged = torch.cat(parts, dim=1)
            if merged.shape[1] == expected_len:
                return merged
        return None

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        sam_mask_shape_list,
        max_new_tokens=32,
        return_attention=False,
    ):
        with torch.inference_mode():
            attention_mask = (
                (input_ids != self.config.pad_token_id).long().to(input_ids.device)
            )
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,
                temperature=0.2,
                use_cache=True,
            )
            output_ids = outputs.sequences

            num_tokens_per_image = self.get_vision_tower().num_patches
            num_rows = output_ids.shape[0]

            # The generated sequence has no hidden state for its very last token,
            # hence pad_right=False; the fallback below feeds the full sequence
            # through one extra forward pass and therefore needs pad_right=True.
            pad_right = False
            seg_token_mask = build_special_token_mask(
                output_ids, self.seg_token_idx, num_tokens_per_image, pad_right=pad_right
            )
            if seg_token_mask.sum() == 0:
                dummy_mask = [torch.zeros(1, 1, dtype=torch.int).to(input_ids.device)]
                if return_attention:
                    return output_ids, dummy_mask, [False], [None]
                return output_ids, dummy_mask, [False]

            hidden_states = self._hidden_states_from_generate(
                outputs, seg_token_mask.shape[1]
            )
            if hidden_states is None:
                # Robust fallback: one teacher-forced pass over the generated
                # sequence gives exactly the same states, correctly aligned.
                images_clip_extend = images_clip.expand(
                    num_rows, -1, -1, -1
                ).contiguous()
                rerun = super(LISATForCausalLM, self).forward(
                    images=images_clip_extend,
                    attention_mask=(output_ids != self.config.pad_token_id).long(),
                    input_ids=output_ids,
                    output_hidden_states=True,
                )
                hidden_states = self._last_hidden_state(rerun.hidden_states)
                pad_right = True
                seg_token_mask = build_special_token_mask(
                    output_ids, self.seg_token_idx, num_tokens_per_image, pad_right=True
                )
            con_token_mask = (
                build_special_token_mask(
                    output_ids, self.con_token_idx, num_tokens_per_image, pad_right=pad_right
                )
                if self.con_token_idx is not None
                else None
            )

            hidden_states = hidden_states.to(seg_token_mask.device)
            image_embeddings = self.get_visual_embs(images)
            # LISAt's evaluation convention is one image per conversation row;
            # a single shared image is broadcast so both layouts work.
            if image_embeddings.shape[0] == 1 and num_rows > 1:
                image_embeddings = image_embeddings.expand(num_rows, -1, -1, -1)
            row_to_image = torch.arange(num_rows, device=output_ids.device)
            prompts, attn_maps, _ = self.compute_dia_prompts(
                hidden_states,
                image_embeddings,
                seg_token_mask,
                con_token_mask,
                row_to_image,
            )

            seg_counts = seg_token_mask.int().sum(-1)
            bounds = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=seg_counts.device),
                    seg_counts.cumsum(-1),
                ]
            )
            prompts_per_row: List[Optional[torch.Tensor]] = []
            attn_per_row: List[Optional[torch.Tensor]] = []
            object_presence: List[bool] = []
            for i in range(len(bounds) - 1):
                if seg_counts[i] == 0:
                    prompts_per_row.append(None)
                    attn_per_row.append(None)
                    object_presence.append(False)
                else:
                    prompts_per_row.append(prompts[bounds[i] : bounds[i + 1]])
                    attn_per_row.append(attn_maps[bounds[i] : bounds[i + 1]])
                    object_presence.append(True)

            pred_masks = self.decode_masks(
                prompts_per_row, image_embeddings, sam_mask_shape_list
            )
            output_pred_masks = []
            for i, pred_mask in enumerate(pred_masks):
                if prompts_per_row[i] is not None:
                    pred_mask = (pred_mask[0] > 0).int()
                    if pred_mask.sum() == 0:
                        object_presence[i] = False
                output_pred_masks.append(pred_mask)

        if return_attention:
            return output_ids, output_pred_masks, object_presence, attn_per_row
        return output_ids, output_pred_masks, object_presence


def load_pretrained_model_DIA_LISAT(
    model_path, device_map="auto", device="cuda", **kwargs
):
    """Inference-time loader — the DIA counterpart of ``load_pretrained_model_LISAT``.

    Use this for checkpoints trained with :func:`init_dia_lisat_model`; the
    plain LISAt loader would silently drop the ``[CON]`` token and the adapter.
    """
    kwargs["device_map"] = device_map

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    con_ids = tokenizer("[CON]", add_special_tokens=False).input_ids
    con_token_idx = con_ids[0] if len(con_ids) == 1 else None
    if con_token_idx is None:
        print(
            "[DIA][WARN] '[CON]' is not a single token of this checkpoint's "
            "tokenizer; falling back to using the [SEG] state as concept query."
        )

    model = DIALISATForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        seg_token_idx=seg_token_idx,
        con_token_idx=con_token_idx,
        **kwargs,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if getattr(model.config, "mm_use_im_patch_token", True):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(model.config, "mm_use_im_start_end", False):
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.resize_token_embeddings(len(tokenizer))
    if kwargs.get("training") is True:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=model.dtype)

    context_len = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, vision_tower, context_len


# --------------------------------------------------------------------------- #
# builder — mirrors model/LISAT.py::init_LISAT_model
# --------------------------------------------------------------------------- #
def init_dia_lisat_model(args, model_args):
    """Build tokenizer + DIA-LISAt model + vision tower.

    Drop-in replacement for ``init_LISAT_model``: the only differences are the
    extra ``[CON]`` token, the DIA hyper-parameters and the fact that the DIA
    modules are excluded from LoRA and marked trainable.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        use_fast=False,
        legacy=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    tokenizer.add_tokens("[CON]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.con_token_idx = tokenizer("[CON]", add_special_tokens=False).input_ids[0]
    model_args["seg_token_idx"] = args.seg_token_idx
    model_args["con_token_idx"] = args.con_token_idx

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model = DIALISATForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    model.get_model().initialize_lisat_modules(model.get_model().config)
    model.initialize_dia_modules(model.config)
    dia_fusion = model.model.dia_fusion  # keep the reference across the LoRA wrap
    model.model.dia_adapter.to(dtype=torch_dtype, device=args.local_rank)
    dia_fusion.to(dtype=torch_dtype, device=args.local_rank)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    if args.lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            exclude_list = [
                "visual_model",
                "vision_tower",
                "mm_projector",
                "text_hidden_fcs",
                "dia_adapter",
                "dia_fusion",
            ]
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and not any(x in name for x in exclude_list)
                    and any(x in name for x in lora_target_modules)
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=find_linear_layers(model, args.lora_target_modules.split(",")),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # Warm-start [CON] from [SEG]: both tokens must live in the same region of
    # the embedding space for the LLM to emit them next to each other early on.
    if getattr(args, "init_con_from_seg", True):
        with torch.no_grad():
            input_emb = model.get_input_embeddings().weight
            input_emb[args.con_token_idx] = input_emb[args.seg_token_idx].clone()
            output_emb = model.get_output_embeddings()
            if output_emb is not None:
                output_emb.weight[args.con_token_idx] = output_emb.weight[
                    args.seg_token_idx
                ].clone()

    trainable_parts = [
        "lm_head",
        "embed_tokens",
        "mask_decoder",
        "text_hidden_fcs",
        "dia_adapter",
        "dia_fusion",
    ]
    for n, p in model.named_parameters():
        if any(part in n for part in trainable_parts):
            p.requires_grad = True

    residual_scale = float(dia_fusion.delta[-1].weight.abs().max())
    print(
        f"[DIA] seg_token_idx={args.seg_token_idx} con_token_idx={args.con_token_idx} | "
        f"fusion residual max|W|={residual_scale:.3e} "
        + ("(identity to LISAt, as expected for a fresh run)" if residual_scale == 0.0
           else "(restored from a DIA checkpoint)")
    )
    return tokenizer, model, vision_tower
