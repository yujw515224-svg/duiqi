import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segment_anything.modeling.common import LayerNorm2d


class ContextEvidenceAdapter(nn.Module):
    """Retrieve local visual evidence from SAM image features with a [CON] query."""

    def __init__(self, dim=256, num_heads=8, num_evidence_tokens=1, dropout=0.0):
        super().__init__()
        self.num_evidence_tokens = num_evidence_tokens
        self.query_proj = nn.Linear(dim, dim)
        self.query_offsets = nn.Parameter(torch.zeros(num_evidence_tokens, dim))
        self.kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, con_embeddings, image_embeddings, image_pe=None):
        batch, channels, height, width = image_embeddings.shape
        assert batch == 1, "DIA-LISAt decodes one SAM image embedding at a time."

        image_tokens = image_embeddings.flatten(2).transpose(1, 2)
        visual_tokens = self.kv_norm(image_tokens)

        value = visual_tokens
        key = visual_tokens
        if image_pe is not None:
            pos = image_pe.flatten(2).transpose(1, 2).to(
                device=visual_tokens.device,
                dtype=visual_tokens.dtype,
            )
            key = key + pos

        query = self.query_proj(con_embeddings).unsqueeze(1)
        query = query + self.query_offsets.unsqueeze(0)

        key = key.expand(query.shape[0], -1, -1)
        value = value.expand(query.shape[0], -1, -1)

        evidence_tokens, attn_probs = self.cross_attn(
            query=query,
            key=key,
            value=value,
            need_weights=True,
            average_attn_weights=True,
        )
        evidence_tokens = self.out_norm(evidence_tokens)
        attn_maps = attn_probs.view(
            query.shape[0],
            query.shape[1],
            height,
            width,
        )

        return evidence_tokens, attn_maps


class ExplicitTokenBridge(nn.Module):
    """Keep the SAM sparse prompt near the original LISAt prompt distribution."""

    def __init__(self, dim=256, init_gate=0.02):
        super().__init__()
        if not 0.0 < init_gate < 1.0:
            raise ValueError(f"init_gate must be in (0, 1), got {init_gate}.")

        self.dim = dim
        self.init_gate = float(init_gate)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(
            self.gate[-1].bias,
            math.log(init_gate / (1.0 - init_gate)),
        )

        self.last_gate_mean = None
        self.last_delta_ratio = None

    def forward(self, anchor_embeddings, explicit_embeddings):
        if anchor_embeddings.shape != explicit_embeddings.shape:
            raise RuntimeError(
                "ExplicitTokenBridge shape mismatch: "
                f"anchor={anchor_embeddings.shape}, explicit={explicit_embeddings.shape}."
            )
        if anchor_embeddings.ndim != 2 or anchor_embeddings.shape[-1] != self.dim:
            raise RuntimeError(
                "ExplicitTokenBridge expects [num_prompts, dim], got "
                f"{anchor_embeddings.shape}."
            )

        difference = explicit_embeddings - anchor_embeddings
        gate_input = torch.cat(
            [anchor_embeddings, explicit_embeddings, difference],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(gate_input))
        stable_embeddings = anchor_embeddings + gate * difference

        with torch.no_grad():
            self.last_gate_mean = gate.detach().float().mean()
            self.last_delta_ratio = (
                (stable_embeddings - anchor_embeddings)
                .detach()
                .float()
                .norm(dim=-1)
                / anchor_embeddings.detach().float().norm(dim=-1).clamp_min(1e-6)
            ).mean()

        return stable_embeddings


class DenseEvidencePrompt(nn.Module):
    """Convert concept attention into a residual SAM dense prompt."""

    def __init__(self, dim=256, attn_clip=8.0):
        super().__init__()
        self.dim = dim
        self.attn_clip = float(attn_clip)

        self.in_proj = nn.Conv2d(dim * 2 + 1, dim, kernel_size=1)
        self.norm = LayerNorm2d(dim)
        self.act = nn.GELU()
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.last_delta_ratio = None
        self.last_attention_mean = None
        self.last_attention_max = None

    def forward(self, image_embeddings, evidence_tokens, attn_maps):
        if image_embeddings.ndim != 4 or image_embeddings.shape[0] != 1:
            raise RuntimeError(
                "DenseEvidencePrompt expects one SAM image embedding "
                f"[1, C, H, W], got {image_embeddings.shape}."
            )
        if evidence_tokens.ndim != 3:
            raise RuntimeError(
                f"evidence_tokens must be [P, K, C], got {evidence_tokens.shape}."
            )
        if attn_maps.ndim != 4:
            raise RuntimeError(f"attn_maps must be [P, K, H, W], got {attn_maps.shape}.")

        num_prompts, num_evidence, channels = evidence_tokens.shape
        if channels != self.dim:
            raise RuntimeError(
                f"Evidence dim mismatch: expected {self.dim}, got {channels}."
            )
        if attn_maps.shape[:2] != (num_prompts, num_evidence):
            raise RuntimeError(
                "Evidence/attention shape mismatch: "
                f"evidence={evidence_tokens.shape}, attn={attn_maps.shape}."
            )

        height, width = image_embeddings.shape[-2:]
        if attn_maps.shape[-2:] != (height, width):
            attn_maps = F.interpolate(
                attn_maps.float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=image_embeddings.dtype)

        attention = attn_maps.mean(dim=1, keepdim=True).float()
        attention = attention / attention.mean(
            dim=(-2, -1),
            keepdim=True,
        ).clamp_min(1e-6)
        attention = attention.clamp(min=0.0, max=self.attn_clip)
        attention = attention.to(
            device=image_embeddings.device,
            dtype=image_embeddings.dtype,
        )

        image = image_embeddings.expand(num_prompts, -1, -1, -1)
        evidence = evidence_tokens.mean(dim=1)
        evidence = evidence[:, :, None, None].expand(-1, -1, height, width)

        dense_input = torch.cat(
            [
                image * attention,
                evidence * attention,
                attention,
            ],
            dim=1,
        )
        hidden = self.act(self.norm(self.in_proj(dense_input)))
        dense_delta = self.out_proj(hidden)

        with torch.no_grad():
            self.last_delta_ratio = (
                dense_delta.detach().float().flatten(1).norm(dim=-1)
                / image.detach().float().flatten(1).norm(dim=-1).clamp_min(1e-6)
            ).mean()
            self.last_attention_mean = attention.detach().float().mean()
            self.last_attention_max = attention.detach().float().amax()

        return dense_delta


class EvidenceGuideFusion(nn.Module):
    """Add visual evidence to [SEG] through a zero-initialized residual path."""

    def __init__(self, dim=256, dropout=0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.res_scale = nn.Parameter(torch.zeros(()))
        self.last_gate_mean = None

    def forward(self, seg_embeddings, con_embeddings, evidence_tokens):
        evidence_summary = evidence_tokens.mean(dim=1)
        fused_input = torch.cat(
            [
                seg_embeddings,
                con_embeddings,
                evidence_summary,
            ],
            dim=-1,
        )

        gate = self.gate(fused_input)
        delta = self.fuse(fused_input)
        fused_seg = seg_embeddings + torch.tanh(self.res_scale) * gate * delta

        self.last_gate_mean = gate.detach().mean()
        return fused_seg.unsqueeze(1)


def attention_alignment_loss(attn_maps, gt_masks, eps=1e-6):
    """KL(target || attention) after area-downsampling the GT mask."""
    attn = attn_maps.mean(dim=1)

    target = F.interpolate(
        gt_masks.unsqueeze(1).float(),
        size=attn.shape[-2:],
        mode="area",
    ).squeeze(1)

    target = target.flatten(1)
    attn = attn.flatten(1)

    valid = target.sum(dim=-1) > eps
    if not valid.any():
        return attn.sum() * 0.0

    target = target[valid]
    attn = attn[valid]

    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    attn = attn.clamp_min(eps)
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(eps)

    return F.kl_div(attn.log(), target, reduction="batchmean")
