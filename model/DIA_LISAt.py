import torch
import torch.nn as nn
import torch.nn.functional as F


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
