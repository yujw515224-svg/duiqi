import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.DIA_LISAt import (  # noqa: E402
    BoundedDenseEvidencePrompt,
    ExplicitRoleAdapter,
)
from model.LISAT import LISATForCausalLM, _validate_dia_structure  # noqa: E402


def _assert_raises(fn, exc_type=RuntimeError):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__}.")


def _peaked_attention(prompts=2, height=4, width=4, scale=5.0):
    return torch.softmax(
        torch.randn(prompts, 1, height * width) * scale,
        dim=-1,
    ).view(prompts, 1, height, width)


def test_role_adapter_starts_exactly_from_anchor():
    torch.manual_seed(0)
    module = ExplicitRoleAdapter(dim=8, hidden_dim=8, max_delta_ratio=0.05)
    anchor = torch.randn(3, 8)
    explicit = torch.randn(3, 8)

    output = module(anchor, explicit)

    assert torch.equal(output, anchor)
    assert module.last_delta_ratio.item() == 0.0


def test_role_adapter_never_exceeds_norm_cap():
    torch.manual_seed(0)
    module = ExplicitRoleAdapter(dim=8, hidden_dim=8, max_delta_ratio=0.05)
    nn.init.normal_(module.role_out.weight, std=10.0)
    nn.init.normal_(module.role_out.bias, std=10.0)
    anchor = torch.randn(4, 8)
    explicit = torch.randn(4, 8)

    output = module(anchor, explicit)
    ratio = (
        (output - anchor).float().norm(dim=-1)
        / anchor.float().norm(dim=-1).clamp_min(1e-6)
    )

    assert torch.all(ratio <= 0.05001)
    assert module.last_delta_ratio.item() <= 0.05001


def test_role_out_gets_gradient_on_first_step_and_upstream_after_step():
    torch.manual_seed(0)
    module = ExplicitRoleAdapter(dim=8, hidden_dim=8)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    anchor = torch.randn(2, 8, requires_grad=True)
    explicit = torch.randn(2, 8, requires_grad=True)
    target = torch.randn(2, 8)

    F.mse_loss(module(anchor, explicit), target).backward()
    assert module.role_out.weight.grad is not None
    assert module.role_out.weight.grad.abs().sum().item() > 0

    optimizer.step()
    optimizer.zero_grad()
    anchor = torch.randn(2, 8, requires_grad=True)
    explicit = torch.randn(2, 8, requires_grad=True)
    target = torch.randn(2, 8)

    F.mse_loss(module(anchor, explicit), target).backward()
    assert explicit.grad is not None
    assert explicit.grad.abs().sum().item() > 0
    assert module.role_in.weight.grad is not None
    assert module.role_in.weight.grad.abs().sum().item() > 0


def test_bounded_dense_starts_from_zero():
    module = BoundedDenseEvidencePrompt(dim=8)
    image = torch.randn(1, 8, 4, 4)
    evidence = torch.randn(2, 1, 8)
    attn = torch.softmax(torch.randn(2, 1, 16), dim=-1).view(2, 1, 4, 4)

    delta = module(image, evidence, attn)

    assert delta.shape == (2, 8, 4, 4)
    assert torch.count_nonzero(delta).item() == 0
    assert module.last_delta_ratio.item() == 0.0


def test_uniform_attention_produces_zero_spatial_residual():
    torch.manual_seed(0)
    module = BoundedDenseEvidencePrompt(dim=8)
    nn.init.normal_(module.out_proj.weight, std=0.2)
    nn.init.normal_(module.norm.weight, std=0.2)
    nn.init.normal_(module.norm.bias, std=0.2)

    image = torch.randn(1, 8, 4, 4)
    evidence = torch.randn(2, 1, 8)
    attn = torch.full((2, 1, 4, 4), 1.0 / 16.0)

    delta = module(image, evidence, attn)

    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)
    assert module.last_confidence_mean.item() < 1e-6
    assert module.last_relative_attention_abs_mean.item() < 1e-6


def test_bounded_dense_never_exceeds_norm_cap():
    torch.manual_seed(0)
    module = BoundedDenseEvidencePrompt(dim=8, max_delta_ratio=0.10)
    nn.init.normal_(module.out_proj.weight, std=10.0)
    image = torch.randn(1, 8, 4, 4)
    evidence = torch.randn(3, 1, 8)
    attn = _peaked_attention(prompts=3)

    delta = module(image, evidence, attn)
    image_ref = image.expand(3, -1, -1, -1)
    ratio = (
        delta.float().flatten(1).norm(dim=-1)
        / image_ref.float().flatten(1).norm(dim=-1).clamp_min(1e-6)
    )

    assert torch.all(ratio <= 0.10001)
    assert module.last_delta_ratio.item() <= 0.10001


def test_dense_confidence_tracks_normalized_entropy():
    torch.manual_seed(0)
    module = BoundedDenseEvidencePrompt(dim=8)
    image = torch.randn(1, 8, 4, 4)
    evidence = torch.randn(1, 1, 8)

    uniform = torch.full((1, 1, 4, 4), 1.0 / 16.0)
    module(image, evidence, uniform)
    uniform_conf = module.last_confidence_mean.item()
    uniform_entropy = module.last_normalized_entropy.item()

    peaked = torch.zeros(1, 1, 4, 4)
    peaked[:, :, 1, 2] = 1.0
    module(image, evidence, peaked)
    peaked_conf = module.last_confidence_mean.item()
    peaked_entropy = module.last_normalized_entropy.item()

    assert 0.0 <= uniform_conf <= 1.0
    assert 0.0 <= peaked_conf <= 1.0
    assert 0.0 <= uniform_entropy <= 1.0
    assert 0.0 <= peaked_entropy <= 1.0
    assert uniform_conf < 1e-6
    assert peaked_conf > uniform_conf
    assert peaked_entropy < uniform_entropy


def test_bounded_dense_out_proj_gets_gradient_on_first_step():
    torch.manual_seed(0)
    module = BoundedDenseEvidencePrompt(dim=8)
    image = torch.randn(1, 8, 4, 4)
    evidence = torch.randn(2, 1, 8, requires_grad=True)
    attn = _peaked_attention(prompts=2, scale=3.0).requires_grad_()

    delta = module(image, evidence, attn)
    target = torch.randn_like(delta)
    F.mse_loss(delta, target).backward()

    assert module.out_proj.weight.grad is not None
    assert module.out_proj.weight.grad.abs().sum().item() > 0


def test_bounded_dense_keeps_attention_and_evidence_in_graph_after_unfreeze():
    torch.manual_seed(0)
    module = BoundedDenseEvidencePrompt(dim=8)
    nn.init.normal_(module.out_proj.weight, std=0.01)
    image = torch.randn(1, 8, 4, 4)
    evidence = torch.randn(2, 1, 8, requires_grad=True)
    attn = _peaked_attention(prompts=2, scale=3.0).requires_grad_()

    module(image, evidence, attn).square().mean().backward()

    assert evidence.grad is not None
    assert evidence.grad.abs().sum().item() > 0
    assert attn.grad is not None
    assert attn.grad.abs().sum().item() > 0


def _fake_model(explicit=True, mode="bounded_sparse_dense"):
    model = object.__new__(LISATForCausalLM)
    model.seg_token_idx = 20
    model.con_token_idx = 10
    model.use_dia = True
    model.explicit_con_in_conversation = explicit
    model.dia_fusion_mode = mode

    class FakeVisionTower:
        num_patches = 3

    model.get_vision_tower = lambda: FakeVisionTower()
    return model


def test_bounded_sparse_dense_requires_strict_adjacent_token_masks():
    model = _fake_model(explicit=True, mode="bounded_sparse_dense")
    input_ids = torch.tensor([[1, 10, 20, 2, 0], [1, 2, 0, 0, 0]])
    seg_mask, con_mask = model.build_dia_token_masks(input_ids, hidden_len=8)

    assert seg_mask.shape == con_mask.shape == (2, 8)
    assert seg_mask[0].sum().item() == 1
    assert con_mask[0].sum().item() == 1
    assert con_mask[0].nonzero().item() + 1 == seg_mask[0].nonzero().item()

    only_seg = torch.tensor([[1, 20, 2, 0]])
    only_con = torch.tensor([[1, 10, 2, 0]])
    reversed_pair = torch.tensor([[1, 20, 10, 2]])
    non_adjacent = torch.tensor([[1, 10, 99, 20]])
    _assert_raises(lambda: model.build_dia_token_masks(only_seg, hidden_len=8))
    _assert_raises(lambda: model.build_dia_token_masks(only_con, hidden_len=8))
    _assert_raises(lambda: model.build_dia_token_masks(reversed_pair, hidden_len=8))
    _assert_raises(lambda: model.build_dia_token_masks(non_adjacent, hidden_len=8))


def test_bounded_sparse_dense_rejects_missing_or_mismatched_anchor_before_sam():
    model = _fake_model(explicit=True, mode="bounded_sparse_dense")
    seg_embeddings = [torch.randn(2, 8)]
    con_embeddings = [torch.randn(2, 8)]
    image_embeddings = torch.randn(1, 8, 4, 4)
    sam_mask_shape_list = [((8, 8), (8, 8))]

    _assert_raises(
        lambda: LISATForCausalLM.generate_pred_masks(
            model,
            seg_embeddings,
            con_embeddings,
            image_embeddings,
            sam_mask_shape_list,
            anchor_embeddings=None,
        )
    )

    bad_anchor = [torch.randn(1, 8)]
    _assert_raises(
        lambda: LISATForCausalLM.generate_pred_masks(
            model,
            seg_embeddings,
            con_embeddings,
            image_embeddings,
            sam_mask_shape_list,
            anchor_embeddings=bad_anchor,
        )
    )


def test_validate_bounded_sparse_dense_structure_accepts_zero_initialized_heads():
    class FakeAttention:
        dropout = 0.0

    class FakeAdapter:
        num_evidence_tokens = 1
        cross_attn = FakeAttention()

    class FakeConfig:
        dia_fusion_mode = "bounded_sparse_dense"
        role_max_delta_ratio = 0.05
        dense_max_delta_ratio = 0.10
        dense_confidence_power = 0.5

    class FakeBase:
        config = FakeConfig()
        context_adapter = FakeAdapter()
        explicit_role_adapter = ExplicitRoleAdapter(dim=8, hidden_dim=8)
        bounded_dense_evidence_prompt = BoundedDenseEvidencePrompt(dim=8)

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
