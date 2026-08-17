# -*- coding: utf-8 -*-
"""Lightweight building blocks for DIA-LISAt.

DIA = Decoupled Image-text Alignment.

The whole proposal only touches the *token-to-mask* path of LISA/LISAt:

    [CON] hidden state  ---(cross-attention over dense SAM features)---> evidence e
    [SEG] hidden state  ---(text_hidden_fcs, unchanged LISAt projector)-> prompt p
    z = Fuse(p, e)      ---> SAM prompt encoder / mask decoder (unchanged)

Nothing in the MLLM backbone, the vision encoder or the SAM decoder is modified,
so the module count added here is ~2M parameters.

This file is backbone agnostic: it only depends on torch, which makes it easy to
unit-test on CPU without any checkpoint (see tests/test_dia_modules.py).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ConceptToEvidenceAdapter",
    "EvidenceGuidedFusion",
    "attention_alignment_loss",
    "attention_mass_in_mask",
    "build_special_token_mask",
    "compute_dia_prompts",
    "decode_masks_with_sam",
    "pair_concept_to_seg",
    "rows_to_image_index",
    "split_by_token_offset",
]


# --------------------------------------------------------------------------- #
# Token bookkeeping
# --------------------------------------------------------------------------- #
def build_special_token_mask(
    input_ids: torch.Tensor,
    token_idx: int,
    num_tokens_per_image: int,
    pad_right: bool = True,
) -> torch.Tensor:
    """Boolean mask selecting the LLM hidden states that emit ``token_idx``.

    This reproduces exactly the convention used by LISA / LISAt:

    * ``input_ids[:, 1:]`` drops BOS, which shifts every position by one, i.e.
      the mask selects the hidden state *that predicts* the special token.
    * ``num_tokens_per_image - 1`` zeros are prepended because the single image
      placeholder token is expanded into ``num_tokens_per_image`` visual tokens
      inside ``LlavaLlamaModel.forward``.
    * one trailing zero keeps the mask length equal to the hidden-state length
      when the full sequence (prompt + answer) is fed in one forward pass.

    ``pad_right=False`` is used on the ``generate()`` path, where the hidden
    state of the very last produced token does not exist.
    """
    mask = input_ids[:, 1:] == token_idx
    pads = [
        torch.zeros(
            mask.shape[0],
            num_tokens_per_image - 1,
            dtype=mask.dtype,
            device=mask.device,
        ),
        mask,
    ]
    if pad_right:
        pads.append(
            torch.zeros(mask.shape[0], 1, dtype=mask.dtype, device=mask.device)
        )
    return torch.cat(pads, dim=1)


def pair_concept_to_seg(
    seg_mask: torch.Tensor,
    con_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pair every ``[SEG]`` occurrence with the ``[CON]`` that introduced it.

    Rule: a ``[SEG]`` is paired with the *closest preceding* ``[CON]`` in the
    same conversation row. When a row contains no usable ``[CON]`` (e.g. the
    model skipped it at inference time, or the answer template was not rewritten)
    the ``[SEG]`` hidden state is used as its own concept query, which makes the
    model degrade gracefully to plain LISAt behaviour instead of crashing.

    Args:
        seg_mask: ``[B, L]`` bool mask produced by :func:`build_special_token_mask`.
        con_mask: ``[B, L]`` bool mask for ``[CON]``, or ``None``.

    Returns:
        ``(seg_flat_idx, con_flat_idx, has_con)`` with shape ``[n_seg_total]``.
        Indices are flat positions into ``hidden.reshape(-1, D)`` and follow the
        same row-major order as ``hidden[seg_mask]``, so they can be zipped with
        the projected ``[SEG]`` embeddings without any re-sorting.
    """
    device = seg_mask.device
    n_rows, length = seg_mask.shape

    seg_flat: List[int] = []
    con_flat: List[int] = []
    has_con: List[bool] = []

    for row in range(n_rows):
        seg_pos = torch.nonzero(seg_mask[row], as_tuple=False).flatten().tolist()
        if not seg_pos:
            continue
        if con_mask is None:
            con_pos: List[int] = []
        else:
            con_pos = torch.nonzero(con_mask[row], as_tuple=False).flatten().tolist()

        for sp in seg_pos:
            before = [cp for cp in con_pos if cp < sp]
            if before:
                con_flat.append(row * length + before[-1])
                has_con.append(True)
            else:
                # graceful fallback: concept query := the [SEG] state itself
                con_flat.append(row * length + sp)
                has_con.append(False)
            seg_flat.append(row * length + sp)

    if not seg_flat:
        empty_long = torch.zeros(0, dtype=torch.long, device=device)
        empty_bool = torch.zeros(0, dtype=torch.bool, device=device)
        return empty_long, empty_long, empty_bool

    return (
        torch.as_tensor(seg_flat, dtype=torch.long, device=device),
        torch.as_tensor(con_flat, dtype=torch.long, device=device),
        torch.as_tensor(has_con, dtype=torch.bool, device=device),
    )


# --------------------------------------------------------------------------- #
# Concept -> Evidence adapter (the cross-attention block of the figure)
# --------------------------------------------------------------------------- #
class ConceptToEvidenceAdapter(nn.Module):
    """Cross-attention from the ``[CON]`` concept token to dense image features.

    Query  : ``h_CON`` (LLM hidden state, ``llm_dim``)
    Key/Val: dense SAM image features ``F`` (``visual_dim x H x W``)

    Returns the aggregated *evidence token* ``e`` and the spatial attention map
    ``A`` (a probability distribution over the ``H*W`` locations), which is the
    quantity supervised by :func:`attention_alignment_loss`.
    """

    def __init__(
        self,
        llm_dim: int,
        visual_dim: int = 256,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_dense_pe: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_dense_pe = use_dense_pe

        self.q_norm = nn.LayerNorm(llm_dim)
        self.kv_norm = nn.LayerNorm(visual_dim)
        self.q_proj = nn.Linear(llm_dim, embed_dim)
        self.k_proj = nn.Linear(visual_dim, embed_dim)
        self.v_proj = nn.Linear(visual_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.reset_dia_parameters()

    def reset_dia_parameters(self) -> None:
        """Re-apply the intended initialisation.

        ``PreTrainedModel`` re-initialises every module that is missing from the
        checkpoint, which would silently overwrite our init, so the model class
        routes ``_init_weights`` back here.
        """
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(
        self,
        concept_hidden: torch.Tensor,
        image_features: torch.Tensor,
        image_pe: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            concept_hidden: ``[N, llm_dim]`` one row per ``[SEG]`` to be decoded.
            image_features: ``[N, C, H, W]`` (or ``[C, H, W]``, broadcast to N)
                dense SAM features of the corresponding image.
            image_pe: optional ``[1, C, H, W]`` SAM positional encoding, added to
                the keys exactly like SAM's own two-way transformer does.

        Returns:
            ``evidence`` ``[N, embed_dim]``, ``attn`` ``[N, H, W]`` (fp32, sums to
            1 over the spatial axes) and a dict of scalar diagnostics.
        """
        if image_features.dim() == 3:
            image_features = image_features.unsqueeze(0).expand(
                concept_hidden.shape[0], -1, -1, -1
            )
        if image_features.shape[0] != concept_hidden.shape[0]:
            raise ValueError(
                "image_features and concept_hidden must agree on the batch dim: "
                f"{image_features.shape[0]} vs {concept_hidden.shape[0]}"
            )

        n, _, height, width = image_features.shape
        num_pixels = height * width
        dtype = self.q_proj.weight.dtype

        tokens = image_features.to(dtype).flatten(2).transpose(1, 2)  # [N, HW, C]
        if self.use_dense_pe and image_pe is not None:
            pe = image_pe.to(dtype)
            if pe.dim() == 4:
                pe = pe.flatten(2).transpose(1, 2)  # [1, HW, C]
            tokens = tokens + pe
        tokens = self.kv_norm(tokens)

        query = self.q_proj(self.q_norm(concept_hidden.to(dtype)))
        query = query.view(n, self.num_heads, self.head_dim)
        keys = self.k_proj(tokens).view(n, num_pixels, self.num_heads, self.head_dim)
        values = self.v_proj(tokens).view(n, num_pixels, self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)      # [N, heads, HW, head_dim]
        values = values.permute(0, 2, 1, 3)

        logits = torch.einsum("nhd,nhkd->nhk", query, keys) * self.scale
        attn = torch.softmax(logits.float(), dim=-1)  # softmax in fp32 for bf16 runs
        context = torch.einsum(
            "nhk,nhkd->nhd", self.attn_drop(attn).to(dtype), values
        ).reshape(n, self.embed_dim)

        evidence = self.out_norm(self.out_proj(context))
        attn_map = attn.mean(dim=1).view(n, height, width)  # still a distribution

        with torch.no_grad():
            entropy = -(attn.clamp_min(1e-12).log() * attn).sum(-1).mean()
            peak = attn_map.flatten(1).max(dim=-1).values.mean()
        return evidence, attn_map, {"attn_entropy": entropy, "attn_peak": peak}


# --------------------------------------------------------------------------- #
# Evidence-guided fusion (the "Fuse" block of the figure)
# --------------------------------------------------------------------------- #
class EvidenceGuidedFusion(nn.Module):
    """Fuse the ``[SEG]`` prompt embedding with the aligned visual evidence.

    ``z = p + clip(g * MLP([p; e]))``

    The last layer of the delta MLP is zero-initialised, therefore ``z == p``
    at step 0: training starts *exactly* at the LISAt baseline and the adapter
    can only earn its influence through the gradient of the mask loss. The
    optional residual cap keeps ``||z - p|| <= max_delta_ratio * ||p||`` so the
    SAM decoder never receives an out-of-distribution prompt.
    """

    def __init__(
        self,
        prompt_dim: int = 256,
        evidence_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        max_delta_ratio: float = 0.5,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_delta_ratio = float(max_delta_ratio)
        self.gate_bias_init = float(gate_bias_init)

        self.norm_p = nn.LayerNorm(prompt_dim)
        self.norm_e = nn.LayerNorm(evidence_dim)
        in_dim = prompt_dim + evidence_dim
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_dim),
        )
        self.delta = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prompt_dim),
        )
        self.reset_dia_parameters()

    def reset_dia_parameters(self) -> None:
        """Zero-init the residual branch so that ``z == p`` at step 0.

        This is the property the whole training recipe rests on, and
        ``PreTrainedModel`` would otherwise re-initialise the module the first
        time it is loaded from a checkpoint that has no DIA weights.
        """
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        nn.init.constant_(self.gate[-1].bias, self.gate_bias_init)

    def forward(
        self, prompt: torch.Tensor, evidence: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """``prompt``/``evidence``: ``[N, prompt_dim]`` / ``[N, evidence_dim]``."""
        out_dtype = prompt.dtype
        dtype = self.delta[0].weight.dtype
        p = prompt.to(dtype)
        e = evidence.to(dtype)

        feats = torch.cat([self.norm_p(p), self.norm_e(e)], dim=-1)
        gate = torch.sigmoid(self.gate(feats))
        delta = gate * self.delta(feats)

        p_norm = p.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        d_norm = delta.norm(dim=-1, keepdim=True)
        if self.max_delta_ratio > 0:
            scale = torch.clamp(
                self.max_delta_ratio * p_norm / d_norm.clamp_min(1e-6), max=1.0
            )
            delta = delta * scale
            d_norm = d_norm * scale

        z = p + delta
        stats = {
            "gate": gate.detach().mean(),
            "delta_ratio": (d_norm / p_norm).detach().mean(),
        }
        return z.to(out_dtype), stats


# --------------------------------------------------------------------------- #
# Supervision on the concept -> evidence attention
# --------------------------------------------------------------------------- #
def _downsample_masks(gt_masks: torch.Tensor, size: Tuple[int, int], mode: str):
    gt = gt_masks.float().unsqueeze(1)
    if mode == "area":
        out = F.interpolate(gt, size=size, mode="area")
    else:  # "max" keeps tiny remote-sensing targets alive at 64x64
        out = F.adaptive_max_pool2d(gt, size)
    return out.squeeze(1)


def attention_alignment_loss(
    attn_maps: torch.Tensor,
    gt_masks: torch.Tensor,
    mode: str = "mass",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, int]:
    """Push the ``[CON]`` attention mass onto the ground-truth region.

    Args:
        attn_maps: ``[N, H, W]`` attention distributions (rows sum to 1).
        gt_masks: ``[N, Ho, Wo]`` binary masks in the *SAM output* resolution.
        mode: ``"mass"`` (default) minimises ``-log`` of the attention mass that
            falls inside the target, which is scale free and therefore stable for
            the very small objects that dominate GRES. ``"kl"`` matches the whole
            distribution against the area-normalised mask.

    Returns:
        ``(loss, n_valid)``. Samples whose mask is empty (negative referring
        samples) are skipped and do not contribute to ``n_valid``.
    """
    if attn_maps.numel() == 0 or gt_masks.numel() == 0:
        return attn_maps.sum() * 0.0, 0
    if attn_maps.shape[0] != gt_masks.shape[0]:
        raise ValueError(
            f"attention/mask count mismatch: {attn_maps.shape[0]} vs {gt_masks.shape[0]}"
        )

    attn = attn_maps.float()
    height, width = attn.shape[-2:]
    target = _downsample_masks(
        gt_masks.to(attn.device), (height, width), "area" if mode == "kl" else "max"
    )

    valid = target.flatten(1).sum(-1) > 0
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return attn.sum() * 0.0, 0

    attn = attn[valid].flatten(1)
    target = target[valid].flatten(1)

    if mode == "mass":
        inside = (target > 0).float()
        mass = (attn * inside).sum(-1).clamp(eps, 1.0)
        loss = -torch.log(mass)
    elif mode == "kl":
        t = target / target.sum(-1, keepdim=True).clamp_min(eps)
        loss = (t * (t.clamp_min(eps).log() - attn.clamp_min(eps).log())).sum(-1)
    else:
        raise ValueError(f"unknown attention alignment mode: {mode}")

    return loss.mean(), n_valid


def rows_to_image_index(num_rows: int, offset: torch.Tensor) -> torch.Tensor:
    """Map every conversation row of the batch to the index of its image.

    ``offset`` is the cumulative conversation count produced by ``collate_fn``:
    rows ``[offset[i], offset[i+1])`` all describe image ``i``.
    """
    bounds = offset.tolist()
    row_to_image = torch.zeros(num_rows, dtype=torch.long, device=offset.device)
    for image_idx in range(len(bounds) - 1):
        row_to_image[bounds[image_idx] : bounds[image_idx + 1]] = image_idx
    return row_to_image


def split_by_token_offset(
    values: torch.Tensor, token_mask: torch.Tensor, offset: torch.Tensor
) -> List[torch.Tensor]:
    """Regroup per-``[SEG]`` rows (row-major order) into one chunk per image."""
    counts = token_mask.int().sum(-1)
    bounds = counts.cumsum(-1)
    bounds = torch.cat(
        [torch.zeros(1, dtype=bounds.dtype, device=bounds.device), bounds]
    )
    bounds = bounds[offset.to(bounds.device)]
    return [values[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]


def compute_dia_prompts(
    hidden_states: torch.Tensor,
    image_embeddings: torch.Tensor,
    seg_token_mask: torch.Tensor,
    con_token_mask: Optional[torch.Tensor],
    row_to_image: torch.Tensor,
    text_hidden_fc: nn.Module,
    adapter: "ConceptToEvidenceAdapter",
    fusion: "EvidenceGuidedFusion",
    image_pe: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Run the full DIA path for every ``[SEG]`` of a batch.

    Args:
        hidden_states: ``[B, L, D]`` last-layer LLM states.
        image_embeddings: ``[n_images, C, H, W]`` dense SAM features.
        seg_token_mask / con_token_mask: ``[B, L]`` masks from
            :func:`build_special_token_mask`.
        row_to_image: ``[B]`` from :func:`rows_to_image_index`.
        text_hidden_fc: LISAt's ``text_hidden_fcs[0]`` projector (unchanged).

    Returns:
        ``z`` ``[n_seg, out_dim]`` fused SAM prompts, ``attn`` ``[n_seg, H, W]``
        and scalar diagnostics. Rows follow the same order as
        ``hidden_states[seg_token_mask]``.
    """
    seq_len = seg_token_mask.shape[1]
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    seg_idx, con_idx, has_con = pair_concept_to_seg(seg_token_mask, con_token_mask)

    # [SEG] -> SAM prompt space through the *unchanged* LISAt projector.
    seg_prompts = text_hidden_fc(flat_hidden[seg_idx])
    concept_hidden = flat_hidden[con_idx]

    image_index = row_to_image.to(seg_idx.device)[
        torch.div(seg_idx, seq_len, rounding_mode="floor")
    ]
    evidence, attn_maps, attn_stats = adapter(
        concept_hidden, image_embeddings[image_index], image_pe=image_pe
    )
    z, fusion_stats = fusion(seg_prompts, evidence)

    stats = {**attn_stats, **fusion_stats}
    stats["con_hit_rate"] = (
        has_con.float().mean() if has_con.numel() else seg_prompts.new_zeros(())
    )
    return z, attn_maps, stats


def decode_masks_with_sam(
    visual_model: nn.Module,
    prompt_embeddings: Sequence[Optional[torch.Tensor]],
    image_embeddings: torch.Tensor,
    sam_mask_shape_list: Sequence[tuple],
) -> List[torch.Tensor]:
    """SAM prompt-encoder + mask-decoder, one call per image.

    Byte-for-byte the behaviour of LISAt's ``generate_pred_masks``; kept here so
    the DIA path stays independent from local edits to ``model/LISAT.py``.
    """
    pred_masks: List[torch.Tensor] = []
    for i, prompt in enumerate(prompt_embeddings):
        input_size, original_size = sam_mask_shape_list[i][0], sam_mask_shape_list[i][1]
        input_size = (int(input_size[0]), int(input_size[1]))
        original_size = (int(original_size[0]), int(original_size[1]))

        if prompt is None:
            pred_masks.append(
                torch.zeros(original_size, device=image_embeddings.device).int()
            )
            continue
        if original_size[0] <= 0 or original_size[1] <= 0:
            pred_masks.append(
                torch.empty(0, *input_size, dtype=prompt.dtype, device=prompt.device)
            )
            continue

        sparse_embeddings, dense_embeddings = visual_model.prompt_encoder(
            points=None, boxes=None, masks=None, text_embeds=prompt.unsqueeze(1)
        )
        sparse_embeddings = sparse_embeddings.to(prompt.dtype)
        low_res_masks, _ = visual_model.mask_decoder(
            image_embeddings=image_embeddings[i].unsqueeze(0),
            image_pe=visual_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        pred_mask = visual_model.postprocess_masks(
            low_res_masks, input_size=input_size, original_size=original_size
        )
        pred_masks.append(pred_mask[:, 0])
    return pred_masks


@torch.no_grad()
def attention_mass_in_mask(
    attn_maps: torch.Tensor, gt_masks: torch.Tensor
) -> Tuple[torch.Tensor, int]:
    """Diagnostic: fraction of attention mass inside the target (higher = better)."""
    if attn_maps.numel() == 0 or gt_masks.numel() == 0:
        return attn_maps.new_zeros(()), 0
    attn = attn_maps.float()
    height, width = attn.shape[-2:]
    target = _downsample_masks(gt_masks.to(attn.device), (height, width), "max")
    valid = target.flatten(1).sum(-1) > 0
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return attn.new_zeros(()), 0
    mass = (attn[valid].flatten(1) * (target[valid].flatten(1) > 0).float()).sum(-1)
    return mass.mean(), n_valid
