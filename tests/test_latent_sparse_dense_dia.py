import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.DIA_LISAt import (  # noqa: E402
    BoundedDenseEvidencePrompt,
    ContextEvidenceAdapter,
    EvidenceVisualBottleneck,
    LatentSparseEvidenceFusion,
)
from model.LISAT import compute_dia_loss_components, _validate_dia_structure  # noqa: E402


def _peaked_attention(prompts=2, evidence=4, height=4, width=4):
    logits = torch.randn(prompts, evidence, height * width) * 3.0
    return torch.softmax(logits, dim=-1).view(prompts, evidence, height, width)


def test_latent_sparse_zero_evidence_is_identity():
    torch.manual_seed(0)
    fusion = LatentSparseEvidenceFusion(
        dim=8,
        hidden_dim=8,
        max_delta_ratio=0.40,
        delta_gain=3.0,
        target_delta_ratio=0.12,
        init_std=1e-3,
    )
    seg = torch.randn(3, 8)
    evidence = torch.zeros(3, 4, 8)

    output = fusion(seg, evidence)

    assert output.shape == (3, 1, 8)
    assert torch.allclose(output[:, 0], seg, atol=1e-7)


def test_latent_sparse_residual_is_bounded_and_trainable():
    torch.manual_seed(0)
    fusion = LatentSparseEvidenceFusion(
        dim=8,
        hidden_dim=8,
        max_delta_ratio=0.40,
        delta_gain=3.0,
        target_delta_ratio=0.12,
        init_std=1e-2,
    )
    seg = torch.randn(3, 8, requires_grad=True)
    evidence = torch.randn(3, 4, 8, requires_grad=True)

    output = fusion(seg, evidence)
    ratio = (
        (output[:, 0] - seg).float().norm(dim=-1)
        / seg.float().norm(dim=-1).clamp_min(1e-6)
    )
    loss = F.mse_loss(output[:, 0], torch.randn_like(seg)) + fusion.last_usage_loss
    loss.backward()

    assert torch.all(ratio <= 0.40001)
    assert torch.isfinite(fusion.last_usage_loss)
    assert fusion.fusion_out.weight.grad is not None
    assert fusion.fusion_out.weight.grad.abs().sum().item() > 0
    assert evidence.grad is not None
    assert evidence.grad.abs().sum().item() > 0


def test_latent_dense_init_is_bounded_and_responds_to_peaked_attention():
    torch.manual_seed(0)
    dense = BoundedDenseEvidencePrompt(
        dim=8,
        max_delta_ratio=0.15,
        confidence_power=0.25,
        out_proj_init_std=1e-3,
    )
    image = torch.randn(1, 8, 4, 4)
    evidence = torch.randn(2, 4, 8, requires_grad=True)
    attn = _peaked_attention(prompts=2).requires_grad_()

    delta = dense(image, evidence, attn)
    image_ref = image.expand(2, -1, -1, -1)
    ratio = (
        delta.float().flatten(1).norm(dim=-1)
        / image_ref.float().flatten(1).norm(dim=-1).clamp_min(1e-6)
    )
    delta.square().mean().backward()

    assert delta.shape == (2, 8, 4, 4)
    assert torch.all(ratio <= 0.15001)
    assert dense.last_confidence_mean.item() > 0.0
    assert dense.out_proj.weight.grad is not None
    assert dense.out_proj.weight.grad.abs().sum().item() > 0
    assert evidence.grad is not None
    assert evidence.grad.abs().sum().item() > 0


def test_visual_bottleneck_uniform_attention_is_identity():
    torch.manual_seed(0)
    bottleneck = EvidenceVisualBottleneck(
        dim=8,
        beta=0.30,
        max_delta_ratio=0.20,
        confidence_power=0.25,
        init_std=1e-3,
    )
    image = torch.randn(1, 8, 4, 4)
    evidence = torch.randn(2, 4, 8)
    attn = torch.full((2, 4, 4, 4), 1.0 / 16.0)

    filtered = bottleneck(image, evidence, attn)

    assert filtered.shape == image.shape
    assert torch.allclose(filtered, image, atol=1e-6)
    assert bottleneck.last_confidence_mean.item() < 1e-6
    assert bottleneck.last_total_delta_ratio.item() < 1e-6


def test_visual_bottleneck_peaked_attention_is_bounded_and_trainable():
    torch.manual_seed(0)
    bottleneck = EvidenceVisualBottleneck(
        dim=8,
        beta=0.30,
        max_delta_ratio=0.20,
        confidence_power=0.25,
        init_std=1e-3,
    )
    image = torch.randn(1, 8, 4, 4, requires_grad=True)
    evidence = torch.randn(2, 4, 8, requires_grad=True)
    attn = _peaked_attention(prompts=2).detach().requires_grad_()

    filtered = bottleneck(image, evidence, attn)
    filtered.square().mean().backward()

    assert filtered.shape == image.shape
    assert bottleneck.last_residual_delta_ratio.item() <= 0.20001
    assert bottleneck.last_confidence_mean.item() > 0.0
    assert bottleneck.out_proj.weight.grad is not None
    assert bottleneck.out_proj.weight.grad.abs().sum().item() > 0
    assert evidence.grad is not None
    assert evidence.grad.abs().sum().item() > 0
    assert attn.grad is not None
    assert attn.grad.abs().sum().item() > 0


def test_latent_sparse_dense_backward_smoke():
    torch.manual_seed(0)
    adapter = ContextEvidenceAdapter(
        dim=8,
        num_heads=2,
        num_evidence_tokens=4,
        dropout=0.0,
    )
    fusion = LatentSparseEvidenceFusion(
        dim=8,
        hidden_dim=8,
        max_delta_ratio=0.40,
        delta_gain=3.0,
        target_delta_ratio=0.12,
        init_std=1e-2,
    )
    dense = BoundedDenseEvidencePrompt(
        dim=8,
        max_delta_ratio=0.15,
        confidence_power=0.25,
        out_proj_init_std=1e-3,
    )
    bottleneck = EvidenceVisualBottleneck(
        dim=8,
        beta=0.30,
        max_delta_ratio=0.20,
        confidence_power=0.25,
        init_std=1e-3,
    )

    seg = torch.randn(2, 8, requires_grad=True)
    con = torch.randn(2, 8, requires_grad=True)
    image = torch.randn(1, 8, 4, 4, requires_grad=True)
    image_pe = torch.randn(1, 8, 4, 4)
    gt = (torch.rand(2, 4, 4) > 0.75).float()

    evidence, attn = adapter(con, image, image_pe)
    prompt = fusion(seg, evidence)
    dense_delta = dense(image, evidence, attn)
    filtered_image = bottleneck(image, evidence, attn)
    pred = (
        prompt[:, 0].mean(dim=-1)[:, None, None]
        + dense_delta.mean(dim=1)
        + filtered_image.mean()
    )
    ce_loss = pred.sum() * 0.0
    loss_dict = compute_dia_loss_components(
        ce_loss=ce_loss,
        pred_masks=[pred],
        gt_masks=[gt],
        attn_maps_list=[attn],
        bce_loss_weight=2.0,
        dice_loss_weight=0.5,
        attn_loss_weight=0.05,
        strict_prompt_alignment=True,
    )
    total = loss_dict["loss"] + 0.10 * fusion.last_usage_loss
    total.backward()

    assert torch.isfinite(total)
    assert adapter.query_proj.weight.grad is not None
    assert adapter.query_proj.weight.grad.abs().sum().item() > 0
    assert fusion.fusion_out.weight.grad is not None
    assert dense.out_proj.weight.grad is not None
    assert bottleneck.out_proj.weight.grad is not None


def test_latent_structure_validator_accepts_k4():
    class FakeAttention:
        dropout = 0.0

    class FakeAdapter:
        num_evidence_tokens = 4
        cross_attn = FakeAttention()

    class FakeConfig:
        dia_fusion_mode = "latent_sparse_dense_dia"
        dia_num_evidence_tokens = 4
        dia_attn_dropout = 0.0
        latent_sparse_max_delta_ratio = 0.40
        latent_sparse_delta_gain = 3.0
        evidence_target_delta_ratio = 0.12
        latent_dense_max_delta_ratio = 0.15
        visual_bottleneck_enabled = True
        visual_bottleneck_beta = 0.30
        visual_bottleneck_max_delta_ratio = 0.20
        visual_bottleneck_confidence_power = 0.25

    class FakeBase:
        config = FakeConfig()
        context_adapter = FakeAdapter()
        latent_sparse_fusion = LatentSparseEvidenceFusion(dim=8, hidden_dim=8)
        latent_dense_evidence_prompt = BoundedDenseEvidencePrompt(
            dim=8,
            max_delta_ratio=0.15,
            out_proj_init_std=1e-3,
        )
        evidence_visual_bottleneck = EvidenceVisualBottleneck(
            dim=8,
            beta=0.30,
            max_delta_ratio=0.20,
            confidence_power=0.25,
            init_std=1e-3,
        )

    class FakeModel:
        def get_model(self):
            return FakeBase()

    _validate_dia_structure(FakeModel())


if __name__ == "__main__":
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
