import pytest
import torch

from model.DIA_LISAt import (
    EvidenceGuideFusionV2,
    SharedEvidenceAdapter,
    build_evidence_targets,
    evidence_map_loss,
    prompt_anchor_loss,
)


def test_shared_evidence_adapter_shapes_and_pooling():
    torch.manual_seed(0)
    adapter = SharedEvidenceAdapter(dim=16, num_heads=4, num_evidence_tokens=1)
    con = torch.randn(3, 16, requires_grad=True)
    image = torch.randn(1, 16, 4, 5, requires_grad=True)
    image_pe = torch.randn(1, 16, 4, 5)

    evidence, map_probs, loc_logits = adapter(con, image, image_pe)

    assert evidence.shape == (3, 1, 16)
    assert map_probs.shape == (3, 1, 4, 5)
    assert loc_logits.shape == (3, 4, 5)
    assert torch.all((map_probs > 0.0) & (map_probs < 1.0))

    visual_tokens = adapter.visual_norm(image.flatten(2).transpose(1, 2))
    values = adapter.v_proj(visual_tokens).squeeze(0)
    weights = map_probs.flatten(2).squeeze(1)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    manual = adapter.out_norm(adapter.out_proj(weights.to(values.dtype) @ values))
    torch.testing.assert_close(evidence[:, 0], manual)

    (evidence.sum() + loc_logits.mean()).backward()
    assert con.grad is not None and torch.isfinite(con.grad).all()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_fusion_warmup_is_exact_identity_and_ramp_changes_prompt():
    torch.manual_seed(1)
    fusion = EvidenceGuideFusionV2(
        dim=16,
        max_strength=0.2,
        warmup_steps=5,
        ramp_steps=10,
        gate_floor=0.1,
        init_gate=0.5,
    )
    fusion.train()
    seg = torch.randn(2, 16, requires_grad=True)
    evidence = torch.randn(2, 1, 16, requires_grad=True)

    warmup_output = fusion(seg, evidence, global_step=0)
    torch.testing.assert_close(warmup_output[:, 0], seg)
    assert fusion.last_strength.item() == 0.0

    ramp_output = fusion(seg, evidence, global_step=15)
    assert ramp_output.shape == (2, 1, 16)
    assert not torch.allclose(ramp_output[:, 0], seg)
    assert abs(fusion.last_strength.item() - 0.2) < 1e-6

    with pytest.raises(TypeError):
        fusion(seg_embeddings=seg, con_embeddings=seg, evidence_tokens=evidence)


def test_evidence_map_loss_prefers_aligned_maps_and_backprops():
    mask = torch.zeros(1, 32, 32)
    mask[:, 8:16, 10:18] = 1.0
    _, target_presence, _ = build_evidence_targets(mask, (4, 4))
    good = torch.where(
        target_presence > 0,
        torch.full_like(target_presence, 4.0),
        torch.full_like(target_presence, -4.0),
    ).requires_grad_()
    bad = (-good.detach()).clone().requires_grad_()

    good_loss = evidence_map_loss(good, mask)
    bad_loss = evidence_map_loss(bad, mask)

    assert torch.isfinite(good_loss)
    assert torch.isfinite(bad_loss)
    assert good_loss.item() < bad_loss.item()
    good_loss.backward()
    assert good.grad is not None and torch.isfinite(good.grad).all()


def test_evidence_map_loss_ignores_empty_targets():
    logits = torch.randn(2, 4, 4, requires_grad=True)
    masks = torch.full((2, 16, 16), 255.0)

    loss = evidence_map_loss(logits, masks)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_prompt_anchor_loss_checks_group_shapes():
    seg = [torch.randn(2, 16)]
    anchor = [seg[0].detach().clone()]
    loss = prompt_anchor_loss(seg, anchor)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6

    with pytest.raises(RuntimeError):
        prompt_anchor_loss(seg, [torch.randn(3, 16)])
