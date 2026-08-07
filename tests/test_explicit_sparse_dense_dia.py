import sys
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.DIA_LISAt import DenseEvidencePrompt, ExplicitTokenBridge  # noqa: E402
from model.LISAT import LISATForCausalLM, _validate_dia_structure  # noqa: E402


def _assert_raises(fn, exc_type=RuntimeError):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__}.")


def test_token_bridge_starts_from_two_percent_explicit():
    torch.manual_seed(0)
    bridge = ExplicitTokenBridge(dim=8, init_gate=0.02)
    anchor = torch.randn(3, 8)
    explicit = torch.randn(3, 8)

    output = bridge(anchor, explicit)
    expected = anchor + 0.02 * (explicit - anchor)

    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-5)
    assert abs(bridge.last_gate_mean.item() - 0.02) < 1e-6
    assert bridge.last_delta_ratio.item() > 0.0


def test_token_bridge_gives_gradient_to_explicit_prompt():
    bridge = ExplicitTokenBridge(dim=8, init_gate=0.02)
    anchor = torch.randn(3, 8, requires_grad=True)
    explicit = torch.randn(3, 8, requires_grad=True)

    bridge(anchor, explicit).square().mean().backward()

    assert anchor.grad is not None
    assert explicit.grad is not None
    assert explicit.grad.abs().sum().item() > 0
    assert bridge.gate[-1].bias.grad is not None


def test_sparse_dense_validation_accepts_low_precision_gate_bias():
    class FakeAttention:
        dropout = 0.0

    class FakeAdapter:
        num_evidence_tokens = 1
        cross_attn = FakeAttention()

    class FakeConfig:
        dia_fusion_mode = "sparse_dense"
        token_bridge_init_gate = 0.02

    class FakeBase:
        config = FakeConfig()
        evidence_adapter = FakeAdapter()
        explicit_token_bridge = ExplicitTokenBridge(dim=8, init_gate=0.02).to(
            torch.bfloat16
        )
        dense_evidence_prompt = DenseEvidencePrompt(dim=8)

    class FakeModel:
        def get_model(self):
            return FakeBase()

    _validate_dia_structure(FakeModel())


def test_dense_prompt_is_zero_at_initialization():
    module = DenseEvidencePrompt(dim=8, attn_clip=8.0)
    image = torch.randn(1, 8, 4, 4)
    evidence = torch.randn(3, 1, 8)
    attn = torch.softmax(torch.randn(3, 1, 4 * 4), dim=-1).view(3, 1, 4, 4)

    delta = module(image, evidence, attn)

    assert delta.shape == (3, 8, 4, 4)
    assert torch.count_nonzero(delta).item() == 0
    assert module.last_delta_ratio.item() == 0.0
    assert torch.isfinite(module.last_attention_mean)
    assert torch.isfinite(module.last_attention_max)


def test_dense_prompt_out_projection_gets_gradient():
    module = DenseEvidencePrompt(dim=8, attn_clip=8.0)
    image = torch.randn(1, 8, 4, 4)
    evidence = torch.randn(2, 1, 8, requires_grad=True)
    attn = torch.softmax(torch.randn(2, 1, 4 * 4), dim=-1)
    attn = attn.view(2, 1, 4, 4).requires_grad_()

    delta = module(image, evidence, attn)
    target = torch.randn_like(delta)
    loss = (delta * target).mean()
    loss.backward()

    assert module.out_proj.weight.grad is not None
    assert module.out_proj.weight.grad.abs().sum().item() > 0


def test_dense_prompt_keeps_attention_in_graph_after_unfreeze():
    module = DenseEvidencePrompt(dim=8, attn_clip=8.0)
    nn.init.normal_(module.out_proj.weight, mean=0.0, std=0.01)
    image = torch.randn(1, 8, 4, 4)
    evidence = torch.randn(2, 1, 8, requires_grad=True)
    attn = torch.softmax(torch.randn(2, 1, 4 * 4), dim=-1)
    attn = attn.view(2, 1, 4, 4).requires_grad_()

    module(image, evidence, attn).square().mean().backward()

    assert attn.grad is not None
    assert attn.grad.abs().sum().item() > 0
    assert evidence.grad is not None
    assert evidence.grad.abs().sum().item() > 0


def _fake_model(explicit=True, mode="sparse_dense"):
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


def test_sparse_dense_requires_strict_adjacent_token_masks():
    model = _fake_model(explicit=True, mode="sparse_dense")
    input_ids = torch.tensor([[1, 10, 20, 2, 0], [1, 2, 0, 0, 0]])
    seg_mask, con_mask = model.build_dia_token_masks(input_ids, hidden_len=8)

    assert seg_mask.shape == con_mask.shape == (2, 8)
    assert seg_mask[0].sum().item() == 1
    assert con_mask[0].sum().item() == 1
    assert not torch.equal(seg_mask, con_mask)

    only_seg = torch.tensor([[1, 20, 2, 0]])
    reversed_pair = torch.tensor([[1, 20, 10, 2]])
    non_adjacent = torch.tensor([[1, 10, 99, 20]])
    _assert_raises(lambda: model.build_dia_token_masks(only_seg, hidden_len=8))
    _assert_raises(lambda: model.build_dia_token_masks(reversed_pair, hidden_len=8))
    _assert_raises(lambda: model.build_dia_token_masks(non_adjacent, hidden_len=8))


def test_sparse_dense_rejects_missing_or_mismatched_anchor_before_sam():
    model = _fake_model(explicit=True, mode="sparse_dense")
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


if __name__ == "__main__":
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
