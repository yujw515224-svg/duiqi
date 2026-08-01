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


class LisatMetaModel(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)

        self.initialize_lisat_modules(self.config)

    def initialize_lisat_modules(self, config):
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
        self.con_hidden_fcs = nn.ModuleList([build_text_project()])
        self.context_adapter = ContextEvidenceAdapter(out_dim, config.dia_num_heads, config.dia_num_evidence_tokens)
        self.evidence_fusion = EvidenceGuideFusion(out_dim)
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


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

        self.con_token_idx = kwargs.pop("con_token_idx")
        self.attn_loss_weight = kwargs.pop("attn_loss_weight", 0.1)
        config.dia_num_heads = kwargs.pop("dia_num_heads", getattr(config, "dia_num_heads", 8))
        config.dia_num_evidence_tokens = kwargs.pop(
            "dia_num_evidence_tokens",
            getattr(config, "dia_num_evidence_tokens", 4),
        )
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

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

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
        """
            用 DIA-LISA prompt 生成预测 mask。

            seg_embeddings:
                List[Tensor]
                每张图对应的 [SEG] embeddings。

            con_embeddings:
                List[Tensor]
                每张图对应的 [CON] embeddings。

            image_embeddings:
                [B, 256, 64, 64]
                SAM image encoder 输出。

            返回:
                pred_masks: List[Tensor]
                attn_maps_list: List[Tensor]
        """
        multimask_output = False
        pred_masks = []
        attn_maps_list = []
        for i in range(len(seg_embeddings)):
            seg_i = seg_embeddings[i]
            con_i = con_embeddings[i]

            input_size, original_size = sam_mask_shape_list[i]
            input_size = (int(input_size[0]), int(input_size[1]))
            original_size = (int(original_size[0]), int(original_size[1]))
            # 如果模型没有生成 [SEG]，就没有办法做 mask prompt  
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
        # 如果只有一个 [CON]，但生成了多个 [SEG]，
        # 就让同一个 concept query 服务多个 mask prompt
            if con_i.shape[0] == 1 and seg_i.shape[0] > 1:
                con_i = con_i.expand(seg_i.shape[0], -1)
        # 如果完全没有 [CON]，说明生成不符合 DIA-LISA 格式。
        # 这里用 [SEG] 作为 fallback，保证推理不直接崩
            if con_i.shape[0] == 0:
                con_i = seg_i
            
            num_prompts = min(seg_i.shape[0], con_i.shape[0])
            seg_i = seg_i[:num_prompts]
            con_i = con_i[:num_prompts]

            decoder_dtype = next(self.model.visual_model.mask_decoder.parameters()).dtype
            decoder_device = image_embeddings.device

            seg_i = seg_i.to(device=decoder_device, dtype=decoder_dtype)
            con_i = con_i.to(device=decoder_device, dtype=decoder_dtype)

            image_i = image_embeddings[i].unsqueeze(0).to(dtype=decoder_dtype)
            # SAM dense positional encoding，用于让 evidence adapter 感知空间位置。
            image_pe = self.model.visual_model.prompt_encoder.get_dense_pe().to(
                device=decoder_device, dtype=decoder_dtype
            )
            # [CON] 从 SAM image feature 里检索 evidence tokens
            evidence_tokens, attn_maps = self.model.context_adapter(
                con_embeddings=con_i,
                image_embeddings=image_i,
                image_pe=image_pe,
            )
            # 融合 [SEG]、[CON]、evidence，得到 SAM text prompt tokens
            prompt_tokens = self.model.evidence_fusion(
                seg_embeddings=seg_i,
                con_embeddings=con_i,
                evidence_tokens=evidence_tokens,
            )

            # prompt_tokens: [N, 1 + K, 256]
            # SAM prompt_encoder 会把它拼进 sparse prompt embeddings。
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
        return pred_masks, attn_maps_list


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
        con_token_mask =  self.build_shifted_token_mask(
            input_ids=input_ids,
            token_idx=self.con_token_idx,
            hidden_len=hidden_len,
        )
         # [SEG] 投影到 SAM prompt embedding 空间。
        seg_flat = self.model.text_hidden_fcs[0](last_hidden_state)[seg_token_mask]
        # [CON] 单独投影，因为它承担 evidence retrieval 的 query 角色
        con_flat = self.model.con_hidden_fcs[0](last_hidden_state)[con_token_mask]
        # 从 conversation-level 重新分组回 image-level。
        seg_embeddings = self.split_embeddings_by_offset(seg_flat, seg_token_mask, offset)
        con_embeddings = self.split_embeddings_by_offset(con_flat, con_token_mask, offset)

        image_embeddings = self.get_visual_embs(images)

        # DIA-LISA mask decoding。
        pred_masks, attn_maps_list = self.generate_pred_masks(
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
        ce_loss = model_output.loss * self.ce_loss_weight

        mask_bce_loss = 0.0
        mask_dice_loss = 0.0
        num_masks = 0
        attn_loss = ce_loss.new_tensor(0.0)
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            assert gt_mask.shape[0] == pred_mask.shape[0], (
                f"Mismatch: gt_mask {gt_mask.shape}, pred_mask {pred_mask.shape}"
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            )
            # 用 GT mask 监督 [CON] 的 visual evidence attention。
            attn_maps = attn_maps_list[batch_idx]
            if attn_maps is not None:
                attn_count = min(attn_maps.shape[0], gt_mask.shape[0])
                if attn_count > 0:
                    attn_loss = attn_loss + (
                        attention_alignment_loss(
                            attn_maps[:attn_count],
                            gt_mask[:attn_count].to(attn_maps.device),
                        )
                        * attn_count
                    )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        attn_loss = attn_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        # 最终总 loss:
        # CE loss 训练语言输出；
        # mask loss 训练最终分割；
        # attention loss 训练 [CON] 找视觉证据
        total_loss = ce_loss + mask_loss + self.attn_loss_weight * attn_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "attn_loss": attn_loss,
        }

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
        with torch.inference_mode():
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

            if self.con_token_idx is not None:
                con_token_mask = self.build_shifted_token_mask(
                    input_ids=output_ids,
                    token_idx=self.con_token_idx,
                    hidden_len=hidden_len,
                )
            else:
                con_token_mask = torch.zeros_like(seg_token_mask)

            seg_token_counts = seg_token_mask.int().sum(-1)
            object_presence = [count.item() > 0 for count in seg_token_counts]

            seg_flat = self.model.text_hidden_fcs[0](last_hidden_state)[seg_token_mask]
            con_flat = self.model.con_hidden_fcs[0](last_hidden_state)[con_token_mask]

            # evaluate() receives one generated conversation per image/question,
            # so each row is its own image-level group.
            offset = torch.arange(
                output_ids.shape[0] + 1,
                dtype=torch.long,
                device=output_ids.device,
            )

            seg_embeddings = self.split_embeddings_by_offset(seg_flat, seg_token_mask, offset)
            con_embeddings = self.split_embeddings_by_offset(con_flat, con_token_mask, offset)

            image_embeddings = self.get_visual_embs(images)

            pred_masks, _ = self.generate_pred_masks(
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

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens(["[SEG]", "[CON]"])

    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    con_token_idx = tokenizer("[CON]", add_special_tokens=False).input_ids[0]

    model = LISATForCausalLM.from_pretrained(
        model_path, 
        low_cpu_mem_usage=True, 
        seg_token_idx=seg_token_idx,
        con_token_idx=con_token_idx, 
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


def init_LISAT_model(args, model_args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        use_fast=False,
        legacy=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens(["[SEG]", "[CON]"])
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

    model = LISATForCausalLM.from_pretrained(
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

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Configure LoRA if applicable
    if args.lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            exclude_list = ["visual_model", "vision_tower", "mm_projector", "text_hidden_fcs", "con_hidden_fcs", "context_adapter", "evidence_fusion"]
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

    model.resize_token_embeddings(len(tokenizer))

    # Make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    trainable_parts = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "con_hidden_fcs", "context_adapter", "evidence_fusion",]
    for n, p in model.named_parameters():
        if any(part in n for part in trainable_parts):
            p.requires_grad = True

    return tokenizer, model, vision_tower