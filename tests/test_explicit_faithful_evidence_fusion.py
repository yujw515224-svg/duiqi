import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.DIA_LISAt import (  # noqa: E402
    ContextEvidenceAdapter,
    EvidenceGuideFusion,
    FaithfulEvidenceFusion,
)
from model.LISAT import (  # noqa: E402
    LISATForCausalLM,
    _validate_dia_structure,
    compute_dia_loss_components,
)


def _assert_raises(fn, exc_type=RuntimeError):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__}.")


def test_faithful_fusion_starts_from_exact_seg_prompt():
    torch.manual_seed(0)
    module = FaithfulEvidenceFusion(dim=8, hidden_dim=16)
    seg = torch.randn(4, 8)
    evidence = torch.randn(4, 1, 8)

    output = module(seg, evidence)

    assert output.shape == (4, 1, 8)
    assert torch.equal(output[:, 0], seg)
    assert module.last_delta_ratio.max().item() == 0.0


def test_faithful_fusion_residual_is_smoothly_bounded():
    torch.manual_seed(0)
    module = FaithfulEvidenceFusion(dim=8, hidden_dim=16, max_delta_ratio=0.15)
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, std=5.0)

    seg = torch.randn(6, 8)
    evidence = torch.randn(6, 1, 8)
    output = module(seg, evidence)[:, 0]
    ratio = (
        (output - seg).float().norm(dim=-1)
        / seg.float().norm(dim=-1).clamp_min(1e-6)
    )

    assert torch.all(ratio <= 0.15001)
    assert module.last_delta_ratio.max().item() <= 0.15001


def test_faithful_fusion_delta_gain_amplifies_but_keeps_cap():
    torch.manual_seed(0)
    base = FaithfulEvidenceFusion(
        dim=8,
        hidden_dim=16,
        max_delta_ratio=0.90,
        delta_gain=1.0,
    )
    strong = FaithfulEvidenceFusion(
        dim=8,
        hidden_dim=16,
        max_delta_ratio=0.90,
        delta_gain=3.0,
    )
    for layer in base.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, std=0.05)
    strong.load_state_dict(base.state_dict())

    seg = torch.randn(8, 8)
    evidence = torch.randn(8, 1, 8)
    base_output = base(seg, evidence)[:, 0]
    strong_output = strong(seg, evidence)[:, 0]
    base_ratio = (
        (base_output - seg).float().norm(dim=-1)
        / seg.float().norm(dim=-1).clamp_min(1e-6)
    )
    strong_ratio = (
        (strong_output - seg).float().norm(dim=-1)
        / seg.float().norm(dim=-1).clamp_min(1e-6)
    )

    assert strong_ratio.mean().item() > base_ratio.mean().item() * 2.0
    assert strong_ratio.max().item() <= 0.90001


def test_faithful_fusion_zero_evidence_keeps_identity_after_randomization():
    torch.manual_seed(0)
    module = FaithfulEvidenceFusion(dim=8, hidden_dim=16, delta_gain=3.0)
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, std=3.0)

    seg = torch.randn(5, 8)
    evidence = torch.zeros(5, 1, 8)
    output = module(seg, evidence)

    assert torch.allclose(output[:, 0], seg, atol=1e-7, rtol=0.0)


def test_faithful_fusion_out_gets_gradient_on_first_step():
    torch.manual_seed(0)
    module = FaithfulEvidenceFusion(dim=8, hidden_dim=16)
    seg = torch.randn(4, 8, requires_grad=True)
    evidence = torch.randn(4, 1, 8, requires_grad=True)
    target = torch.randn(4, 1, 8)

    (module(seg, evidence) * target).mean().backward()

    assert module.fusion_out.weight.grad is not None
    assert module.fusion_out.weight.grad.abs().sum().item() > 0


def test_context_adapter_key_only_position_encoding_and_attention_grad():
    torch.manual_seed(0)
    adapter = ContextEvidenceAdapter(
        dim=4,
        num_heads=2,
        num_evidence_tokens=1,
        dropout=0.0,
    )
    captured = {}

    def fake_attn(query, key, value, need_weights=True, average_attn_weights=True):
        captured["key"] = key
        captured["value"] = value
        probs = torch.softmax(query @ key.transpose(-1, -2), dim=-1)
        evidence = probs @ value
        return evidence, probs

    adapter.cross_attn.forward = fake_attn
    con = torch.randn(2, 4, requires_grad=True)
    image = torch.randn(1, 4, 2, 2, requires_grad=True)
    image_pe = torch.randn(1, 4, 2, 2)

    evidence, attn = adapter(con, image, image_pe=image_pe)
    pos = image_pe.flatten(2).transpose(1, 2).expand(2, -1, -1)

    assert evidence.shape == (2, 1, 4)
    assert attn.shape == (2, 1, 2, 2)
    assert attn.requires_grad
    assert torch.allclose(
        captured["key"] - captured["value"],
        pos.to(dtype=captured["key"].dtype),
        atol=1e-6,
    )
    assert torch.allclose(
        attn.flatten(-2).sum(-1),
        torch.ones(2, 1),
        atol=1e-6,
    )


def _fake_model(explicit=True, mode="faithful_evidence_fusion"):
    model = object.__new__(LISATForCausalLM)
    model.seg_token_idx = 20
    model.con_token_idx = 10
    model.use_dia = True
    model.explicit_con_in_conversation = explicit
    model.dia_fusion_mode = mode
    model.dia_bypass_fusion = False

    class FakeVisionTower:
        num_patches = 3

    model.get_vision_tower = lambda: FakeVisionTower()
    return model


def test_faithful_requires_strict_adjacent_shifted_masks():
    model = _fake_model()
    input_ids = torch.tensor([[1, 10, 20, 2, 0], [1, 2, 0, 0, 0]])
    seg_mask, con_mask = model.build_dia_token_masks(input_ids, hidden_len=8)

    assert seg_mask.shape == con_mask.shape == (2, 8)
    assert not torch.equal(seg_mask, con_mask)
    assert con_mask[0].nonzero().item() + 1 == seg_mask[0].nonzero().item()

    _assert_raises(
        lambda: model.build_dia_token_masks(torch.tensor([[1, 20, 2, 0]]), 8)
    )
    _assert_raises(
        lambda: model.build_dia_token_masks(torch.tensor([[1, 10, 2, 0]]), 8)
    )
    _assert_raises(
        lambda: model.build_dia_token_masks(torch.tensor([[1, 20, 10, 2]]), 8)
    )
    _assert_raises(
        lambda: model.build_dia_token_masks(torch.tensor([[1, 10, 99, 20]]), 8)
    )


def test_faithful_loss_requires_exact_attention_gt_count():
    ce = torch.tensor(0.1, requires_grad=True)
    pred_masks = [torch.zeros(2, 4, 4)]
    gt_masks = [torch.zeros(2, 4, 4)]
    attn_maps_list = [torch.full((1, 1, 4, 4), 1.0 / 16.0)]

    _assert_raises(
        lambda: compute_dia_loss_components(
            ce,
            pred_masks,
            gt_masks,
            attn_maps_list,
            bce_loss_weight=2.0,
            dice_loss_weight=0.5,
            attn_loss_weight=0.02,
            strict_prompt_alignment=True,
        )
    )
    result = compute_dia_loss_components(
        ce,
        pred_masks,
        gt_masks,
        attn_maps_list,
        bce_loss_weight=2.0,
        dice_loss_weight=0.5,
        attn_loss_weight=0.02,
        strict_prompt_alignment=False,
    )
    assert torch.isfinite(result["loss"])


class _FakePromptEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.calls = 0
        self.last_masks = "unset"

    def get_dense_pe(self):
        return torch.zeros(1, self.dim, 4, 4)

    def forward(self, points=None, boxes=None, masks=None, text_embeds=None):
        self.calls += 1
        self.last_masks = masks
        prompt_count = text_embeds.shape[0]
        sparse = text_embeds
        dense = text_embeds.new_zeros(prompt_count, self.dim, 4, 4)
        return sparse, dense


class _FakeMaskDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = 0
        self.dim = dim

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output=False,
    ):
        self.calls += 1
        prompt_count = sparse_prompt_embeddings.shape[0]
        masks = sparse_prompt_embeddings.new_ones(prompt_count, 1, 4, 4)
        return masks * self.weight, None


class _FakeVisualModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.prompt_encoder = _FakePromptEncoder(dim)
        self.mask_decoder = _FakeMaskDecoder(dim)

    def postprocess_masks(self, low_res_masks, input_size, original_size):
        return low_res_masks


class _FakeAdapter(nn.Module):
    def forward(self, con_embeddings, image_embeddings, image_pe=None):
        prompts, dim = con_embeddings.shape
        evidence = torch.ones(prompts, 1, dim, device=con_embeddings.device)
        attn = torch.full(
            (prompts, 1, 4, 4),
            1.0 / 16.0,
            device=con_embeddings.device,
        )
        return evidence, attn


def test_faithful_decode_rejects_mismatched_con_seg_counts():
    model = _fake_model()
    model.model = SimpleNamespace(
        context_adapter=_FakeAdapter(),
        faithful_evidence_fusion=FaithfulEvidenceFusion(dim=8, hidden_dim=16),
        visual_model=_FakeVisualModel(dim=8),
    )

    _assert_raises(
        lambda: LISATForCausalLM.generate_pred_masks(
            model,
            seg_embeddings=[torch.randn(2, 8)],
            con_embeddings=[torch.randn(1, 8)],
            image_embeddings=torch.randn(1, 8, 4, 4),
            sam_mask_shape_list=[((4, 4), (4, 4))],
        )
    )


def test_faithful_decode_calls_sam_once_with_masks_none():
    model = _fake_model()
    visual_model = _FakeVisualModel(dim=8)
    model.model = SimpleNamespace(
        context_adapter=_FakeAdapter(),
        faithful_evidence_fusion=FaithfulEvidenceFusion(dim=8, hidden_dim=16),
        visual_model=visual_model,
    )

    pred_masks, attn_maps, stats = LISATForCausalLM.generate_pred_masks(
        model,
        seg_embeddings=[torch.randn(2, 8)],
        con_embeddings=[torch.randn(2, 8)],
        image_embeddings=torch.randn(1, 8, 4, 4),
        sam_mask_shape_list=[((4, 4), (4, 4))],
    )

    assert pred_masks[0].shape == (2, 4, 4)
    assert attn_maps[0].shape == (2, 1, 4, 4)
    assert visual_model.prompt_encoder.calls == 1
    assert visual_model.prompt_encoder.last_masks is None
    assert visual_model.mask_decoder.calls == 1
    assert torch.stack(stats["sam_prompt_encoder_calls"]).sum().item() == 1
    assert torch.stack(stats["sam_mask_decoder_calls"]).sum().item() == 1


def test_validate_faithful_structure_accepts_required_zero_out():
    class FakeAttention:
        dropout = 0.0

    class FakeAdapter:
        num_evidence_tokens = 1
        cross_attn = FakeAttention()

    class FakeConfig:
        dia_fusion_mode = "faithful_evidence_fusion"
        faithful_fusion_hidden_dim = 16
        faithful_max_delta_ratio = 0.15
        faithful_delta_gain = 2.5

    class FakeBase:
        config = FakeConfig()
        context_adapter = FakeAdapter()
        faithful_evidence_fusion = FaithfulEvidenceFusion(
            dim=8,
            hidden_dim=16,
            delta_gain=2.5,
        )

    class FakeModel:
        def get_model(self):
            return FakeBase()

    _validate_dia_structure(FakeModel())


def test_old_legacy_mode_still_validates_independently():
    class FakeAttention:
        dropout = 0.0

    class FakeAdapter:
        num_evidence_tokens = 1
        cross_attn = FakeAttention()

    class FakeConfig:
        dia_fusion_mode = "legacy"

    class FakeBase:
        config = FakeConfig()
        context_adapter = FakeAdapter()
        evidence_fusion = EvidenceGuideFusion(dim=8)

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
