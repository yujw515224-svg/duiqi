import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segment_anything.modeling.common import LayerNorm2d


def _inverse_sigmoid(prob: float) -> float:
    if not 0.0 < prob < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {prob}")
    return math.log(prob / (1.0 - prob))


class SharedEvidenceAdapter(nn.Module):
    """Use one supervised non-competitive map for localization and pooling."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_evidence_tokens: int = 1,
        loc_bias_init: float = -4.0,
    ):
        super().__init__()
        if num_evidence_tokens != 1:
            raise ValueError("evidence_feedback requires exactly one evidence token")
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.num_evidence_tokens = num_evidence_tokens
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_norm = nn.LayerNorm(dim)
        self.visual_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_norm = nn.LayerNorm(dim)
        self.loc_bias = nn.Parameter(torch.tensor(float(loc_bias_init)))

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.eye_(self.v_proj.weight)
        nn.init.eye_(self.out_proj.weight)

    def forward(self, con_embeddings, image_embeddings, image_pe=None):
        if con_embeddings.ndim != 2 or con_embeddings.shape[-1] != self.dim:
            raise RuntimeError(
                f"con_embeddings must be [P, {self.dim}], got {con_embeddings.shape}"
            )
        if image_embeddings.ndim != 4 or image_embeddings.shape[:2] != (1, self.dim):
            raise RuntimeError(
                f"image_embeddings must be [1, {self.dim}, H, W], "
                f"got {image_embeddings.shape}"
            )

        _, _, height, width = image_embeddings.shape
        num_prompts = con_embeddings.shape[0]
        num_locations = height * width

        image_tokens = image_embeddings.flatten(2).transpose(1, 2)
        visual_tokens = self.visual_norm(image_tokens)

        key_input = visual_tokens
        if image_pe is not None:
            if image_pe.shape[-2:] != (height, width):
                raise RuntimeError(
                    f"image_pe spatial shape {image_pe.shape[-2:]} does not match "
                    f"image embedding shape {(height, width)}"
                )
            pos = image_pe.flatten(2).transpose(1, 2).to(
                device=visual_tokens.device,
                dtype=visual_tokens.dtype,
            )
            key_input = key_input + pos

        q = self.q_proj(self.query_norm(con_embeddings))
        k = self.k_proj(key_input)
        v = self.v_proj(visual_tokens)

        q = q.view(num_prompts, self.num_heads, self.head_dim)
        k = (
            k.view(1, num_locations, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .squeeze(0)
        )

        head_logits = torch.einsum("phd,hnd->phn", q.float(), k.float()) * self.scale
        loc_logits = head_logits.mean(dim=1) + self.loc_bias.float()
        map_probs = torch.sigmoid(loc_logits)

        pool_weights = map_probs / map_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        evidence = torch.einsum(
            "pn,nc->pc",
            pool_weights.to(dtype=v.dtype),
            v.squeeze(0),
        )
        evidence = self.out_norm(self.out_proj(evidence))

        evidence_tokens = evidence.unsqueeze(1)
        map_probs = map_probs.view(num_prompts, 1, height, width)
        loc_logits = loc_logits.view(num_prompts, height, width)
        return evidence_tokens, map_probs, loc_logits


class EvidenceGuideFusionV2(nn.Module):
    """Inject evidence-only residuals with a deterministic warmup/ramp."""

    def __init__(
        self,
        dim: int = 256,
        max_strength: float = 0.15,
        warmup_steps: int = 2000,
        ramp_steps: int = 4000,
        gate_floor: float = 0.1,
        init_gate: float = 0.5,
    ):
        super().__init__()
        if not 0.0 < max_strength <= 1.0:
            raise ValueError("max_strength must be in (0, 1]")
        if warmup_steps < 0 or ramp_steps <= 0:
            raise ValueError("warmup_steps must be >= 0 and ramp_steps must be > 0")
        if not 0.0 <= gate_floor < init_gate < 1.0:
            raise ValueError("expected 0 <= gate_floor < init_gate < 1")

        self.dim = dim
        self.max_strength = float(max_strength)
        self.warmup_steps = int(warmup_steps)
        self.ramp_steps = int(ramp_steps)
        self.gate_floor = float(gate_floor)

        self.seg_norm = nn.LayerNorm(dim)
        self.evidence_norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim * 3, dim)
        self.evidence_proj = nn.Linear(dim, dim, bias=False)

        raw_init_gate = (init_gate - gate_floor) / (1.0 - gate_floor)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, _inverse_sigmoid(raw_init_gate))
        nn.init.eye_(self.evidence_proj.weight)

        self.last_gate_mean = None
        self.last_strength = None
        self.last_delta_ratio = None

    def strength_at(self, global_step, training: bool) -> float:
        if not training:
            return self.max_strength
        if global_step is None:
            raise RuntimeError("training evidence_feedback requires dia_global_step")
        step = int(global_step)
        if step < self.warmup_steps:
            return 0.0
        progress = min(1.0, (step - self.warmup_steps) / float(self.ramp_steps))
        return self.max_strength * progress

    def forward(self, seg_embeddings, evidence_tokens, global_step=None):
        if seg_embeddings.ndim != 2 or seg_embeddings.shape[-1] != self.dim:
            raise RuntimeError(
                f"seg_embeddings must be [P, {self.dim}], got {seg_embeddings.shape}"
            )
        if evidence_tokens.shape != (seg_embeddings.shape[0], 1, self.dim):
            raise RuntimeError(
                f"evidence_tokens must be [P, 1, {self.dim}], got {evidence_tokens.shape}"
            )

        evidence = evidence_tokens[:, 0]
        seg_norm = self.seg_norm(seg_embeddings)
        evidence_norm = self.evidence_norm(evidence)
        gate_input = torch.cat(
            [seg_norm, evidence_norm, seg_norm * evidence_norm],
            dim=-1,
        )
        raw_gate = torch.sigmoid(self.gate(gate_input))
        gate = self.gate_floor + (1.0 - self.gate_floor) * raw_gate

        evidence_delta = self.evidence_proj(evidence_norm)
        strength = self.strength_at(global_step, self.training)
        fused_seg = seg_embeddings + strength * gate * evidence_delta

        with torch.no_grad():
            self.last_gate_mean = gate.detach().float().mean()
            self.last_strength = seg_embeddings.new_tensor(strength).detach()
            self.last_delta_ratio = (
                (fused_seg - seg_embeddings).detach().float().norm(dim=-1)
                / seg_embeddings.detach().float().norm(dim=-1).clamp_min(1e-6)
            ).mean()

        return fused_seg.unsqueeze(1)


def build_evidence_targets(gt_masks, output_size, eps=1e-6):
    if gt_masks.ndim != 3:
        raise RuntimeError(f"gt_masks must be [P, H, W], got {gt_masks.shape}")
    target = gt_masks.float().unsqueeze(1)
    valid = (target != 255).float()
    foreground = ((target > 0.5) & (target != 255)).float()

    valid_fraction = F.adaptive_avg_pool2d(valid, output_size)
    foreground_fraction = F.adaptive_avg_pool2d(foreground, output_size)
    target_area = foreground_fraction / valid_fraction.clamp_min(eps)
    target_area = target_area.clamp(0.0, 1.0)
    target_presence = F.adaptive_max_pool2d(foreground, output_size)
    return (
        target_area.squeeze(1),
        target_presence.squeeze(1),
        valid_fraction.squeeze(1),
    )


def evidence_map_loss(
    loc_logits,
    gt_masks,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    dice_weight: float = 0.5,
    eps: float = 1e-6,
):
    if loc_logits.ndim != 3:
        raise RuntimeError(f"loc_logits must be [P, H, W], got {loc_logits.shape}")
    if loc_logits.shape[0] != gt_masks.shape[0]:
        raise RuntimeError(
            f"evidence/GT prompt mismatch: logits={loc_logits.shape}, gt={gt_masks.shape}"
        )

    target_area, target_presence, valid_weight = build_evidence_targets(
        gt_masks,
        loc_logits.shape[-2:],
        eps=eps,
    )
    positive_prompt = (target_presence * valid_weight).flatten(1).sum(-1) > 0
    if not positive_prompt.any():
        return loc_logits.sum() * 0.0

    logits = loc_logits[positive_prompt].float()
    area = target_area[positive_prompt].float()
    presence = target_presence[positive_prompt].float()
    weight = valid_weight[positive_prompt].float()
    probs = torch.sigmoid(logits)

    bce = F.binary_cross_entropy_with_logits(logits, presence, reduction="none")
    pt = probs * presence + (1.0 - probs) * (1.0 - presence)
    alpha_t = focal_alpha * presence + (1.0 - focal_alpha) * (1.0 - presence)
    focal = alpha_t * (1.0 - pt).pow(focal_gamma) * bce
    focal = (focal * weight).sum() / weight.sum().clamp_min(eps)

    intersection = (probs * area * weight).flatten(1).sum(-1)
    denominator = (
        (probs * weight).flatten(1).sum(-1)
        + (area * weight).flatten(1).sum(-1)
    )
    dice = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    return focal + dice_weight * dice.mean()


def prompt_anchor_loss(seg_embeddings, anchor_embeddings, eps=1e-6):
    if len(seg_embeddings) != len(anchor_embeddings):
        raise RuntimeError("seg/anchor image-group counts differ")

    seg_parts = []
    anchor_parts = []
    for image_idx, (seg_i, anchor_i) in enumerate(zip(seg_embeddings, anchor_embeddings)):
        if seg_i.shape != anchor_i.shape:
            raise RuntimeError(
                f"seg/anchor mismatch at image {image_idx}: "
                f"seg={seg_i.shape}, anchor={anchor_i.shape}"
            )
        if seg_i.shape[0] > 0:
            seg_parts.append(seg_i.float())
            anchor_parts.append(anchor_i.detach().float())

    if not seg_parts:
        if not seg_embeddings:
            raise RuntimeError("seg_embeddings must contain at least one image group")
        return seg_embeddings[0].sum() * 0.0

    seg = torch.cat(seg_parts, dim=0)
    anchor = torch.cat(anchor_parts, dim=0)
    cosine = 1.0 - F.cosine_similarity(seg, anchor, dim=-1).mean()
    seg_norm = seg.norm(dim=-1).clamp_min(eps).log()
    anchor_norm = anchor.norm(dim=-1).clamp_min(eps).log()
    norm = F.smooth_l1_loss(seg_norm, anchor_norm)
    return cosine + 0.1 * norm


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


def _bounded_residual(raw_delta, reference, max_ratio, eps=1e-6):
    """Bound each sample independently relative to its reference norm."""
    if raw_delta.shape[0] != reference.shape[0]:
        raise RuntimeError(
            "Residual/reference batch mismatch: "
            f"delta={raw_delta.shape}, reference={reference.shape}."
        )
    if max_ratio <= 0.0:
        raise ValueError(f"max_ratio must be positive, got {max_ratio}.")

    raw_norm = raw_delta.float().flatten(1).norm(dim=-1)
    ref_norm = reference.float().flatten(1).norm(dim=-1).clamp_min(eps)
    pre_ratio = raw_norm / ref_norm
    bound_scale = torch.clamp(
        max_ratio / pre_ratio.clamp_min(eps),
        max=1.0,
    ).detach()

    view_shape = [raw_delta.shape[0]] + [1] * (raw_delta.ndim - 1)
    bounded_delta = raw_delta * bound_scale.to(
        device=raw_delta.device,
        dtype=raw_delta.dtype,
    ).view(*view_shape)
    post_ratio = bounded_delta.float().flatten(1).norm(dim=-1) / ref_norm
    return bounded_delta, pre_ratio, post_ratio, bound_scale


class ExplicitRoleAdapter(nn.Module):
    """Map explicit [SEG] information to a bounded sparse residual."""

    def __init__(self, dim=256, hidden_dim=256, max_delta_ratio=0.05, eps=1e-6):
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive.")
        if not 0.0 < max_delta_ratio <= 0.10:
            raise ValueError(
                "role max_delta_ratio must be in (0, 0.10], got "
                f"{max_delta_ratio}."
            )

        self.dim = int(dim)
        self.max_delta_ratio = float(max_delta_ratio)
        self.eps = float(eps)

        self.role_norm = nn.LayerNorm(dim * 3)
        self.role_in = nn.Linear(dim * 3, hidden_dim)
        self.act = nn.GELU()
        self.role_out = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.role_out.weight)
        nn.init.zeros_(self.role_out.bias)

        self.last_preclip_ratio = None
        self.last_delta_ratio = None
        self.last_bound_scale = None
        self.last_bound_hit_rate = None

    def forward(self, anchor_embeddings, explicit_embeddings):
        if anchor_embeddings.shape != explicit_embeddings.shape:
            raise RuntimeError(
                "ExplicitRoleAdapter shape mismatch: "
                f"anchor={anchor_embeddings.shape}, explicit={explicit_embeddings.shape}."
            )
        if anchor_embeddings.ndim != 2 or anchor_embeddings.shape[-1] != self.dim:
            raise RuntimeError(
                "ExplicitRoleAdapter expects [num_prompts, dim], got "
                f"{anchor_embeddings.shape}."
            )

        difference = explicit_embeddings - anchor_embeddings
        role_input = torch.cat(
            [anchor_embeddings, explicit_embeddings, difference],
            dim=-1,
        )
        raw_delta = self.role_out(
            self.act(self.role_in(self.role_norm(role_input)))
        )
        role_delta, pre_ratio, post_ratio, bound_scale = _bounded_residual(
            raw_delta,
            anchor_embeddings,
            self.max_delta_ratio,
            self.eps,
        )
        stable_embeddings = anchor_embeddings + role_delta

        with torch.no_grad():
            self.last_preclip_ratio = pre_ratio.detach().mean()
            self.last_delta_ratio = post_ratio.detach().mean()
            self.last_bound_scale = bound_scale.detach().mean()
            self.last_bound_hit_rate = (bound_scale.detach() < 1.0).float().mean()

        return stable_embeddings


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


class BoundedDenseEvidencePrompt(nn.Module):
    """Centered, confidence-gated and norm-bounded SAM dense residual."""

    def __init__(
        self,
        dim=256,
        attn_clip=8.0,
        max_delta_ratio=0.10,
        confidence_power=0.5,
        out_proj_init_std=0.0,
        eps=1e-6,
    ):
        super().__init__()
        if attn_clip <= 1.0:
            raise ValueError("attn_clip must be greater than 1.")
        if not 0.0 < max_delta_ratio <= 0.20:
            raise ValueError("dense max_delta_ratio must be in (0, 0.20].")
        if confidence_power <= 0.0:
            raise ValueError("confidence_power must be positive.")
        if out_proj_init_std < 0.0:
            raise ValueError("out_proj_init_std must be non-negative.")

        self.dim = int(dim)
        self.attn_clip = float(attn_clip)
        self.max_delta_ratio = float(max_delta_ratio)
        self.confidence_power = float(confidence_power)
        self.out_proj_init_std = float(out_proj_init_std)
        self.eps = float(eps)

        self.in_proj = nn.Conv2d(dim * 2 + 1, dim, 1, bias=False)
        self.norm = LayerNorm2d(dim)
        self.act = nn.GELU()
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)
        if self.out_proj_init_std == 0.0:
            nn.init.zeros_(self.out_proj.weight)
        else:
            nn.init.normal_(self.out_proj.weight, std=self.out_proj_init_std)

        self.last_preclip_ratio = None
        self.last_delta_ratio = None
        self.last_bound_scale = None
        self.last_bound_hit_rate = None
        self.last_confidence_mean = None
        self.last_normalized_entropy = None
        self.last_relative_attention_abs_mean = None

    def forward(self, image_embeddings, evidence_tokens, attn_maps):
        if image_embeddings.ndim != 4 or image_embeddings.shape[0] != 1:
            raise RuntimeError(
                f"BoundedDenseEvidencePrompt expects image [1,C,H,W], got {image_embeddings.shape}."
            )
        if evidence_tokens.ndim != 3 or attn_maps.ndim != 4:
            raise RuntimeError(
                "BoundedDenseEvidencePrompt expects evidence [P,K,C] and attention [P,K,H,W]."
            )

        prompts, evidence_count, channels = evidence_tokens.shape
        if channels != self.dim:
            raise RuntimeError(f"Expected evidence dim {self.dim}, got {channels}.")
        if attn_maps.shape[:2] != (prompts, evidence_count):
            raise RuntimeError("Evidence/attention prompt counts differ.")

        _, image_channels, height, width = image_embeddings.shape
        if image_channels != self.dim or height * width <= 1:
            raise RuntimeError(
                "Invalid SAM image embedding shape: "
                f"image={image_embeddings.shape}, expected dim={self.dim}."
            )
        if attn_maps.shape[-2:] != (height, width):
            attn_maps = F.interpolate(
                attn_maps.float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        probability = attn_maps.mean(dim=1, keepdim=True).float()
        probability = probability.clamp_min(0.0)
        probability = probability / probability.sum(
            dim=(-2, -1),
            keepdim=True,
        ).clamp_min(self.eps)

        entropy = -(
            probability * probability.clamp_min(self.eps).log()
        ).sum(dim=(-2, -1))
        normalized_entropy = (
            entropy / math.log(height * width)
        ).clamp(0.0, 1.0)
        confidence = (1.0 - normalized_entropy).clamp(0.0, 1.0).pow(
            self.confidence_power
        )

        relative_attention = probability / probability.mean(
            dim=(-2, -1),
            keepdim=True,
        ).clamp_min(self.eps)
        relative_attention = (relative_attention - 1.0).clamp(
            min=-1.0,
            max=self.attn_clip - 1.0,
        )
        relative_attention = relative_attention.to(
            device=image_embeddings.device,
            dtype=image_embeddings.dtype,
        )

        image = image_embeddings.expand(prompts, -1, -1, -1)
        evidence = evidence_tokens.mean(dim=1).to(
            device=image.device,
            dtype=image.dtype,
        )
        evidence = evidence[:, :, None, None].expand(-1, -1, height, width)
        dense_input = torch.cat(
            [
                image * relative_attention,
                evidence * relative_attention,
                relative_attention,
            ],
            dim=1,
        )
        raw_delta = self.out_proj(self.act(self.norm(self.in_proj(dense_input))))

        centered_delta = raw_delta.float()
        centered_delta = centered_delta - centered_delta.mean(
            dim=(-2, -1),
            keepdim=True,
        )
        centered_delta = centered_delta.to(dtype=raw_delta.dtype)
        bounded_delta, pre_ratio, _, bound_scale = _bounded_residual(
            centered_delta,
            image,
            self.max_delta_ratio,
            self.eps,
        )

        confidence_gate = confidence.detach().to(
            device=bounded_delta.device,
            dtype=bounded_delta.dtype,
        )[:, :, None, None]
        dense_delta = bounded_delta * confidence_gate
        final_ratio = (
            dense_delta.float().flatten(1).norm(dim=-1)
            / image.float().flatten(1).norm(dim=-1).clamp_min(self.eps)
        )

        with torch.no_grad():
            self.last_preclip_ratio = pre_ratio.detach().mean()
            self.last_delta_ratio = final_ratio.detach().mean()
            self.last_bound_scale = bound_scale.detach().mean()
            self.last_bound_hit_rate = (bound_scale.detach() < 1.0).float().mean()
            self.last_confidence_mean = confidence.detach().mean()
            self.last_normalized_entropy = normalized_entropy.detach().mean()
            self.last_relative_attention_abs_mean = (
                relative_attention.detach().float().abs().mean()
            )

        return dense_delta


class EvidenceVisualBottleneck(nn.Module):
    """Use CON attention to suppress irrelevant SAM image features.

    The module keeps the SAM decoder interface unchanged: one image embedding
    enters and one image embedding leaves. CON attention only controls a
    spatial gate and a bounded evidence residual, so [SEG] remains the single
    sparse prompt that asks SAM to decode the final mask.
    """

    def __init__(
        self,
        dim=256,
        beta=0.30,
        attn_clip=8.0,
        max_delta_ratio=0.20,
        confidence_power=0.25,
        init_std=1e-3,
        eps=1e-6,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"visual bottleneck beta must be in (0, 1], got {beta}.")
        if attn_clip <= 1.0:
            raise ValueError("visual bottleneck attn_clip must be greater than 1.")
        if not 0.0 < max_delta_ratio <= 0.50:
            raise ValueError(
                "visual bottleneck max_delta_ratio must be in (0, 0.50], "
                f"got {max_delta_ratio}."
            )
        if confidence_power <= 0.0:
            raise ValueError("visual bottleneck confidence_power must be positive.")
        if init_std < 0.0:
            raise ValueError("visual bottleneck init_std must be non-negative.")

        self.dim = int(dim)
        self.beta = float(beta)
        self.attn_clip = float(attn_clip)
        self.max_delta_ratio = float(max_delta_ratio)
        self.confidence_power = float(confidence_power)
        self.init_std = float(init_std)
        self.eps = float(eps)

        self.evidence_proj = nn.Linear(dim, dim, bias=False)
        self.in_proj = nn.Conv2d(dim * 2 + 1, dim, 1, bias=False)
        self.norm = LayerNorm2d(dim)
        self.act = nn.GELU()
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)
        if self.init_std == 0.0:
            nn.init.zeros_(self.out_proj.weight)
        else:
            nn.init.normal_(self.out_proj.weight, std=self.init_std)

        self.last_gate_mean = None
        self.last_confidence_mean = None
        self.last_image_delta_ratio = None
        self.last_residual_delta_ratio = None
        self.last_total_delta_ratio = None
        self.last_bound_scale = None
        self.last_bound_hit_rate = None

    def _attention_gate(self, attn_maps, height, width):
        if attn_maps.shape[-2:] != (height, width):
            attn_maps = F.interpolate(
                attn_maps.float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        probability = attn_maps.mean(dim=(0, 1), keepdim=True).float()
        probability = probability.clamp_min(0.0)
        probability = probability / probability.sum(
            dim=(-2, -1),
            keepdim=True,
        ).clamp_min(self.eps)

        entropy = -(
            probability * probability.clamp_min(self.eps).log()
        ).sum(dim=(-2, -1))
        normalized_entropy = (
            entropy / math.log(max(height * width, 2))
        ).clamp(0.0, 1.0)
        confidence = (1.0 - normalized_entropy).clamp(0.0, 1.0).pow(
            self.confidence_power
        )

        relative_attention = probability / probability.mean(
            dim=(-2, -1),
            keepdim=True,
        ).clamp_min(self.eps)
        clipped_attention = relative_attention.clamp(
            min=0.0,
            max=self.attn_clip,
        )
        spatial_gate = self.beta + (1.0 - self.beta) * (
            clipped_attention / self.attn_clip
        )
        effective_gate = 1.0 + confidence[:, :, None, None] * (
            spatial_gate - 1.0
        )
        centered_attention = (relative_attention - 1.0).clamp(
            min=-1.0,
            max=self.attn_clip - 1.0,
        )
        return effective_gate, centered_attention, confidence

    def forward(self, image_embeddings, evidence_tokens, attn_maps):
        if image_embeddings.ndim != 4 or image_embeddings.shape[0] != 1:
            raise RuntimeError(
                "EvidenceVisualBottleneck expects one image embedding "
                f"[1,C,H,W], got {tuple(image_embeddings.shape)}."
            )
        if evidence_tokens.ndim != 3 or attn_maps.ndim != 4:
            raise RuntimeError(
                "EvidenceVisualBottleneck expects evidence [P,K,C] and "
                f"attention [P,K,H,W], got {tuple(evidence_tokens.shape)} "
                f"and {tuple(attn_maps.shape)}."
            )
        if evidence_tokens.shape[:2] != attn_maps.shape[:2]:
            raise RuntimeError(
                "EvidenceVisualBottleneck prompt/evidence count mismatch: "
                f"evidence={tuple(evidence_tokens.shape)}, "
                f"attn={tuple(attn_maps.shape)}."
            )

        _, channels, height, width = image_embeddings.shape
        if channels != self.dim:
            raise RuntimeError(
                f"EvidenceVisualBottleneck expected dim={self.dim}, got {channels}."
            )
        if evidence_tokens.shape[-1] != self.dim:
            raise RuntimeError(
                "EvidenceVisualBottleneck evidence dim mismatch: "
                f"expected={self.dim}, got {evidence_tokens.shape[-1]}."
            )

        gate, centered_attention, confidence = self._attention_gate(
            attn_maps,
            height,
            width,
        )
        gate = gate.to(device=image_embeddings.device, dtype=image_embeddings.dtype)
        centered_attention = centered_attention.to(
            device=image_embeddings.device,
            dtype=image_embeddings.dtype,
        )
        confidence = confidence.to(
            device=image_embeddings.device,
            dtype=image_embeddings.dtype,
        )

        gated_image = image_embeddings * gate
        evidence_summary = evidence_tokens.mean(dim=(0, 1), keepdim=False)
        evidence_summary = self.evidence_proj(
            evidence_summary.to(
                device=image_embeddings.device,
                dtype=image_embeddings.dtype,
            )
        )
        evidence_map = evidence_summary[None, :, None, None].expand(
            -1,
            -1,
            height,
            width,
        )
        delta_input = torch.cat(
            [
                gated_image,
                evidence_map * centered_attention,
                centered_attention,
            ],
            dim=1,
        )
        raw_delta = self.out_proj(self.act(self.norm(self.in_proj(delta_input))))
        raw_delta = raw_delta - raw_delta.float().mean(
            dim=(-2, -1),
            keepdim=True,
        ).to(dtype=raw_delta.dtype)

        bounded_delta, _, residual_ratio, bound_scale = _bounded_residual(
            raw_delta,
            image_embeddings,
            self.max_delta_ratio,
            self.eps,
        )
        confidence_gate = confidence[:, :, None, None]
        filtered_image = gated_image + confidence_gate * bounded_delta

        ref_norm = image_embeddings.float().flatten(1).norm(dim=-1).clamp_min(
            self.eps
        )
        image_delta_ratio = (
            (gated_image - image_embeddings).float().flatten(1).norm(dim=-1)
            / ref_norm
        )
        total_delta_ratio = (
            (filtered_image - image_embeddings).float().flatten(1).norm(dim=-1)
            / ref_norm
        )
        with torch.no_grad():
            self.last_gate_mean = gate.detach().float().mean()
            self.last_confidence_mean = confidence.detach().float().mean()
            self.last_image_delta_ratio = image_delta_ratio.detach().mean()
            self.last_residual_delta_ratio = residual_ratio.detach().mean()
            self.last_total_delta_ratio = total_delta_ratio.detach().mean()
            self.last_bound_scale = bound_scale.detach().mean()
            self.last_bound_hit_rate = (bound_scale.detach() < 1.0).float().mean()

        return filtered_image


class FaithfulEvidenceFusion(nn.Module):
    """Fuse [CON]-retrieved evidence into the explicit [SEG] sparse prompt.

    The residual path is evidence-only: if evidence is all zeros, every layer
    before the output also receives zeros, so the module remains an exact
    identity no matter how the linear weights are later updated.
    """

    def __init__(
        self,
        dim=256,
        hidden_dim=256,
        max_delta_ratio=0.15,
        delta_gain=1.0,
        eps=1e-6,
    ):
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive.")
        if max_delta_ratio <= 0.0:
            raise ValueError(
                f"max_delta_ratio must be positive, got {max_delta_ratio}."
            )
        if delta_gain <= 0.0:
            raise ValueError(f"delta_gain must be positive, got {delta_gain}.")

        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.max_delta_ratio = float(max_delta_ratio)
        self.delta_gain = float(delta_gain)
        self.eps = float(eps)

        self.seg_norm = nn.LayerNorm(dim)
        self.evidence_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.evidence_proj = nn.Linear(dim, dim, bias=False)
        self.fusion_in = nn.Linear(dim * 2, hidden_dim, bias=False)
        self.fusion_out = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.fusion_out.weight)

        self.last_raw_delta_ratio = None
        self.last_delta_ratio = None
        self.last_smooth_scale = None

    def forward(self, seg_embeddings, evidence_tokens):
        if seg_embeddings.ndim != 2:
            raise RuntimeError(
                "seg_embeddings must be [num_prompts, dim], got "
                f"{tuple(seg_embeddings.shape)}."
            )
        if evidence_tokens.ndim != 3:
            raise RuntimeError(
                "evidence_tokens must be [num_prompts, K, dim], got "
                f"{tuple(evidence_tokens.shape)}."
            )
        if seg_embeddings.shape[0] != evidence_tokens.shape[0]:
            raise RuntimeError(
                "Prompt/evidence count mismatch: "
                f"seg={seg_embeddings.shape[0]}, "
                f"evidence={evidence_tokens.shape[0]}."
            )
        if (
            seg_embeddings.shape[-1] != self.dim
            or evidence_tokens.shape[-1] != self.dim
        ):
            raise RuntimeError(
                "FaithfulEvidenceFusion dim mismatch: "
                f"seg={tuple(seg_embeddings.shape)}, "
                f"evidence={tuple(evidence_tokens.shape)}, "
                f"expected_dim={self.dim}."
            )

        evidence_summary = evidence_tokens.mean(dim=1)
        seg_feature = self.seg_norm(seg_embeddings)
        evidence_feature = self.evidence_proj(
            self.evidence_norm(evidence_summary)
        )

        interaction = torch.cat(
            [
                evidence_feature,
                seg_feature * evidence_feature,
            ],
            dim=-1,
        )
        raw_delta = self.fusion_out(F.gelu(self.fusion_in(interaction)))
        raw_delta = raw_delta * self.delta_gain

        raw_norm = raw_delta.float().norm(dim=-1)
        seg_norm = seg_embeddings.float().norm(dim=-1).clamp_min(self.eps)
        raw_ratio = raw_norm / seg_norm
        smooth_scale = torch.rsqrt(
            1.0 + (raw_ratio / self.max_delta_ratio).square()
        ).detach()
        evidence_delta = raw_delta * smooth_scale.to(
            device=raw_delta.device,
            dtype=raw_delta.dtype,
        ).unsqueeze(-1)
        delta_ratio = evidence_delta.float().norm(dim=-1) / seg_norm

        with torch.no_grad():
            self.last_raw_delta_ratio = raw_ratio.detach()
            self.last_delta_ratio = delta_ratio.detach()
            self.last_smooth_scale = smooth_scale.detach()

        return (seg_embeddings + evidence_delta).unsqueeze(1)


class DecoupledMaskPrompt(nn.Module):
    """Build a SAM prompt from visual evidence without reading ``[SEG]``.

    ``[SEG]`` is deliberately reduced to a routing/count signal.  Target
    semantics can therefore reach SAM only through the ``[CON]`` query and its
    retrieved visual evidence.  Keeping ``seg_embeddings`` out of this API is
    an architectural guard against restoring the original semantic shortcut.
    """

    def __init__(self, dim=256, hidden_dim=256, max_delta_ratio=0.15):
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive")
        if max_delta_ratio <= 0:
            raise ValueError("max_delta_ratio must be positive")
        self.dim = int(dim)
        self.max_delta_ratio = float(max_delta_ratio)
        self.decoder_anchor = nn.Parameter(torch.empty(1, dim))
        nn.init.normal_(self.decoder_anchor, std=0.02)
        self.register_buffer("anchor_initialized", torch.tensor(False), persistent=True)
        self.evidence_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.evidence_in = nn.Linear(dim, hidden_dim, bias=False)
        self.evidence_out = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.evidence_out.weight)
        self.last_delta_ratio = None

    @torch.no_grad()
    def initialize_anchor_from_teacher(self, teacher_sum, teacher_count):
        """Seed the shared anchor once from a distributed teacher-prompt mean."""
        if bool(self.anchor_initialized.item()) or float(teacher_count.item()) <= 0:
            return
        self.decoder_anchor.copy_((teacher_sum / teacher_count).view(1, self.dim))
        self.anchor_initialized.fill_(True)

    def forward(self, evidence_tokens, num_prompts):
        if evidence_tokens.ndim != 3 or evidence_tokens.shape[-1] != self.dim:
            raise RuntimeError(
                "evidence_tokens must be [P,K,dim], got "
                f"{tuple(evidence_tokens.shape)}"
            )
        if int(num_prompts) != evidence_tokens.shape[0]:
            raise RuntimeError(
                f"prompt/evidence mismatch: {num_prompts} vs "
                f"{evidence_tokens.shape[0]}"
            )
        # Preserve K independently retrieved local evidence tokens.  Collapsing
        # them with a mean would recreate the single-token information bottleneck.
        raw_delta = self.evidence_out(
            F.gelu(self.evidence_in(self.evidence_norm(evidence_tokens)))
        )
        anchor = self.decoder_anchor.to(
            device=raw_delta.device, dtype=raw_delta.dtype
        ).view(1, 1, self.dim).expand_as(raw_delta)
        delta, _, ratio, _ = _bounded_residual(
            raw_delta, anchor, self.max_delta_ratio
        )
        self.last_delta_ratio = ratio.detach()
        return anchor + delta


class EvidencePresenceHead(nn.Module):
    """Predict whether a concept has supporting visual evidence.

    The head consumes only the explicit concept representation and retrieved
    visual evidence; it never reads the instance-specific [SEG] hidden state.
    """

    def __init__(self, dim=256, hidden_dim=256):
        super().__init__()
        self.con_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.evidence_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.input = nn.Linear(dim * 3, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, con_embeddings, evidence_tokens):
        if con_embeddings.ndim != 2 or evidence_tokens.ndim != 3:
            raise RuntimeError("presence inputs must be [N,D] and [N,K,D]")
        evidence = self.evidence_norm(evidence_tokens.mean(dim=1))
        concept = self.con_norm(con_embeddings)
        joint = torch.cat([concept, evidence, concept * evidence], dim=-1)
        return self.output(F.gelu(self.input(joint))).squeeze(-1)


class LatentSparseEvidenceFusion(nn.Module):
    """Aggressively inject latent visual evidence into the SAM sparse prompt.

    This module is intentionally used without explicit [CON] text. The concept
    query comes from the same hidden state that produced the original [SEG]
    prompt, then the retrieved SAM evidence adds a bounded residual to [SEG].
    The residual path is evidence-only, so zero evidence still gives an exact
    identity output.
    """

    def __init__(
        self,
        dim=256,
        hidden_dim=256,
        max_delta_ratio=0.40,
        delta_gain=3.0,
        target_delta_ratio=0.12,
        init_std=1e-3,
        eps=1e-6,
    ):
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive.")
        if not 0.0 < max_delta_ratio <= 0.60:
            raise ValueError(
                "latent sparse max_delta_ratio must be in (0, 0.60], got "
                f"{max_delta_ratio}."
            )
        if delta_gain <= 0.0:
            raise ValueError(f"delta_gain must be positive, got {delta_gain}.")
        if not 0.0 <= target_delta_ratio <= max_delta_ratio:
            raise ValueError(
                "target_delta_ratio must be between 0 and max_delta_ratio."
            )
        if init_std < 0.0:
            raise ValueError("init_std must be non-negative.")

        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.max_delta_ratio = float(max_delta_ratio)
        self.delta_gain = float(delta_gain)
        self.target_delta_ratio = float(target_delta_ratio)
        self.init_std = float(init_std)
        self.eps = float(eps)

        self.seg_norm = nn.LayerNorm(dim)
        self.evidence_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.evidence_proj = nn.Linear(dim, dim, bias=False)
        self.fusion_in = nn.Linear(dim * 2, hidden_dim, bias=False)
        self.fusion_out = nn.Linear(hidden_dim, dim, bias=False)
        if self.init_std == 0.0:
            nn.init.zeros_(self.fusion_out.weight)
        else:
            nn.init.normal_(self.fusion_out.weight, std=self.init_std)

        self.last_raw_delta_ratio = None
        self.last_delta_ratio = None
        self.last_bound_scale = None
        self.last_bound_hit_rate = None
        self.last_usage_loss = None

    def forward(self, seg_embeddings, evidence_tokens):
        if seg_embeddings.ndim != 2:
            raise RuntimeError(
                "seg_embeddings must be [num_prompts, dim], got "
                f"{tuple(seg_embeddings.shape)}."
            )
        if evidence_tokens.ndim != 3:
            raise RuntimeError(
                "evidence_tokens must be [num_prompts, K, dim], got "
                f"{tuple(evidence_tokens.shape)}."
            )
        if seg_embeddings.shape[0] != evidence_tokens.shape[0]:
            raise RuntimeError(
                "Prompt/evidence count mismatch: "
                f"seg={seg_embeddings.shape[0]}, "
                f"evidence={evidence_tokens.shape[0]}."
            )
        if (
            seg_embeddings.shape[-1] != self.dim
            or evidence_tokens.shape[-1] != self.dim
        ):
            raise RuntimeError(
                "LatentSparseEvidenceFusion dim mismatch: "
                f"seg={tuple(seg_embeddings.shape)}, "
                f"evidence={tuple(evidence_tokens.shape)}, "
                f"expected_dim={self.dim}."
            )

        evidence_summary = evidence_tokens.mean(dim=1)
        seg_feature = self.seg_norm(seg_embeddings)
        evidence_feature = self.evidence_proj(
            self.evidence_norm(evidence_summary)
        )
        interaction = torch.cat(
            [
                evidence_feature,
                seg_feature * evidence_feature,
            ],
            dim=-1,
        )

        raw_delta = self.fusion_out(F.gelu(self.fusion_in(interaction)))
        raw_delta = raw_delta * self.delta_gain
        evidence_delta, raw_ratio, delta_ratio, bound_scale = _bounded_residual(
            raw_delta,
            seg_embeddings,
            self.max_delta_ratio,
            self.eps,
        )

        self.last_usage_loss = F.relu(
            self.target_delta_ratio - delta_ratio
        ).mean()
        with torch.no_grad():
            self.last_raw_delta_ratio = raw_ratio.detach()
            self.last_delta_ratio = delta_ratio.detach()
            self.last_bound_scale = bound_scale.detach()
            self.last_bound_hit_rate = (bound_scale.detach() < 1.0).float()

        return (seg_embeddings + evidence_delta).unsqueeze(1)


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
