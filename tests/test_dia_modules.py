# -*- coding: utf-8 -*-
"""CPU unit tests for the DIA-LISAt modules (no checkpoint / GPU needed).

    python tests/test_dia_modules.py     # or: pytest tests/test_dia_modules.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.dia_conversation import (  # noqa: E402
    count_tokens,
    insert_con_tokens,
    validate_con_seg_pairs,
)
from model.dia_modules import (  # noqa: E402
    ConceptToEvidenceAdapter,
    has_meta_parameters,
    EvidenceGuidedFusion,
    attention_alignment_loss,
    attention_mass_in_mask,
    build_special_token_mask,
    pair_concept_to_seg,
)

SEG_ID, CON_ID, NUM_PATCHES = 32000, 32001, 256


# --------------------------------------------------------------------------- #
# token bookkeeping
# --------------------------------------------------------------------------- #
def test_token_mask_length_matches_hidden_states():
    """mask length == LLM hidden length (image token -> num_patches tokens)."""
    seq_len = 40
    input_ids = torch.zeros(2, seq_len, dtype=torch.long)
    mask = build_special_token_mask(input_ids, SEG_ID, NUM_PATCHES)
    assert mask.shape == (2, seq_len - 1 + NUM_PATCHES)

    gen_mask = build_special_token_mask(input_ids, SEG_ID, NUM_PATCHES, pad_right=False)
    assert gen_mask.shape == (2, seq_len - 2 + NUM_PATCHES)


def test_token_mask_selects_the_state_that_emits_the_token():
    """LISA convention: the hidden state *before* [SEG] is the mask embedding."""
    input_ids = torch.zeros(1, 10, dtype=torch.long)
    input_ids[0, 7] = SEG_ID
    mask = build_special_token_mask(input_ids, SEG_ID, NUM_PATCHES)
    (position,) = torch.nonzero(mask[0], as_tuple=False).flatten().tolist()
    # input position 7 lives at hidden index 7 + NUM_PATCHES - 1; we select one
    # step earlier, i.e. the state whose logits produce [SEG].
    assert position == 7 + NUM_PATCHES - 2


def test_pairing_multiple_targets():
    """k-th [SEG] takes the closest preceding [CON], per conversation row."""
    seq_len = 24
    input_ids = torch.zeros(2, seq_len, dtype=torch.long)
    input_ids[0, 5], input_ids[0, 9] = CON_ID, SEG_ID
    input_ids[0, 12], input_ids[0, 16] = CON_ID, SEG_ID
    input_ids[1, 4], input_ids[1, 8] = CON_ID, SEG_ID

    seg_mask = build_special_token_mask(input_ids, SEG_ID, NUM_PATCHES)
    con_mask = build_special_token_mask(input_ids, CON_ID, NUM_PATCHES)
    seg_idx, con_idx, has_con = pair_concept_to_seg(seg_mask, con_mask)

    assert seg_idx.shape[0] == 3 and bool(has_con.all())
    length = seg_mask.shape[1]
    rows = torch.div(seg_idx, length, rounding_mode="floor").tolist()
    assert rows == [0, 0, 1]
    # every concept must sit strictly before its own [SEG] and in the same row
    assert bool((con_idx < seg_idx).all())
    assert torch.div(con_idx, length, rounding_mode="floor").tolist() == rows


def test_pairing_falls_back_when_no_concept_token():
    input_ids = torch.zeros(1, 20, dtype=torch.long)
    input_ids[0, 11] = SEG_ID
    seg_mask = build_special_token_mask(input_ids, SEG_ID, NUM_PATCHES)
    con_mask = build_special_token_mask(input_ids, CON_ID, NUM_PATCHES)

    seg_idx, con_idx, has_con = pair_concept_to_seg(seg_mask, con_mask)
    assert seg_idx.tolist() == con_idx.tolist()  # concept query := [SEG] itself
    assert not bool(has_con.any())

    seg_idx2, con_idx2, has_con2 = pair_concept_to_seg(seg_mask, None)
    assert seg_idx2.tolist() == con_idx2.tolist() and not bool(has_con2.any())


def test_pairing_ignores_concepts_emitted_after_seg():
    input_ids = torch.zeros(1, 20, dtype=torch.long)
    input_ids[0, 6] = SEG_ID
    input_ids[0, 9] = CON_ID  # generated too late -> unusable
    seg_mask = build_special_token_mask(input_ids, SEG_ID, NUM_PATCHES)
    con_mask = build_special_token_mask(input_ids, CON_ID, NUM_PATCHES)
    seg_idx, con_idx, has_con = pair_concept_to_seg(seg_mask, con_mask)
    assert not bool(has_con.any()) and seg_idx.tolist() == con_idx.tolist()


def test_pairing_empty_batch():
    input_ids = torch.zeros(2, 12, dtype=torch.long)
    seg_mask = build_special_token_mask(input_ids, SEG_ID, NUM_PATCHES)
    seg_idx, con_idx, has_con = pair_concept_to_seg(seg_mask, None)
    assert seg_idx.numel() == 0 and con_idx.numel() == 0 and has_con.numel() == 0


# --------------------------------------------------------------------------- #
# concept -> evidence adapter
# --------------------------------------------------------------------------- #
def test_adapter_shapes_and_normalisation():
    torch.manual_seed(0)
    adapter = ConceptToEvidenceAdapter(llm_dim=64, visual_dim=32, embed_dim=32, num_heads=4)
    concept = torch.randn(3, 64)
    features = torch.randn(3, 32, 8, 8)
    evidence, attn, stats = adapter(concept, features)

    assert evidence.shape == (3, 32)
    assert attn.shape == (3, 8, 8)
    assert torch.allclose(attn.flatten(1).sum(-1), torch.ones(3), atol=1e-5)
    assert attn.min() >= 0
    assert "attn_entropy" in stats and "attn_peak" in stats


def test_adapter_accepts_shared_features_and_dense_pe():
    adapter = ConceptToEvidenceAdapter(llm_dim=64, visual_dim=32, embed_dim=32, num_heads=4)
    evidence, attn, _ = adapter(
        torch.randn(2, 64), torch.randn(32, 8, 8), image_pe=torch.randn(1, 32, 8, 8)
    )
    assert evidence.shape == (2, 32) and attn.shape == (2, 8, 8)


def test_adapter_attends_to_the_matching_region():
    """A concept whose query matches one patch must concentrate mass there."""
    torch.manual_seed(0)
    dim = 32
    adapter = ConceptToEvidenceAdapter(
        llm_dim=dim, visual_dim=dim, embed_dim=dim, num_heads=1
    )
    with torch.no_grad():  # identity projections: attention == feature similarity
        adapter.q_proj.weight.copy_(torch.eye(dim))
        adapter.k_proj.weight.copy_(torch.eye(dim))
        adapter.q_proj.bias.zero_()
        adapter.k_proj.bias.zero_()

    features = torch.randn(1, dim, 4, 4)
    concept = features[0, :, 2, 3].unsqueeze(0)  # ask for that exact patch
    _, attn, _ = adapter(concept, features)
    assert attn[0].flatten().argmax().item() == 2 * 4 + 3


# --------------------------------------------------------------------------- #
# evidence-guided fusion
# --------------------------------------------------------------------------- #
def test_fusion_is_identity_at_initialisation():
    """Zero-init residual => DIA starts exactly at the LISAt baseline."""
    torch.manual_seed(0)
    fusion = EvidenceGuidedFusion(prompt_dim=16, evidence_dim=16, hidden_dim=16)
    prompt = torch.randn(5, 16)
    z, stats = fusion(prompt, torch.randn(5, 16))
    assert torch.allclose(z, prompt, atol=1e-6)
    assert float(stats["delta_ratio"]) == 0.0


def test_meta_parameters_are_detected():
    """from_pretrained(low_cpu_mem_usage=True) leaves new modules on meta.

    Those carry no data, so .to(device) raises and they must be rebuilt --
    this check is what tells the model to do so.
    """
    with torch.device("meta"):
        meta_adapter = ConceptToEvidenceAdapter(llm_dim=16, visual_dim=16, embed_dim=16, num_heads=2)
        meta_fusion = EvidenceGuidedFusion(prompt_dim=16, evidence_dim=16, hidden_dim=16)
    assert has_meta_parameters(meta_adapter)
    assert has_meta_parameters(meta_fusion)

    real = ConceptToEvidenceAdapter(llm_dim=16, visual_dim=16, embed_dim=16, num_heads=2)
    assert not has_meta_parameters(real)
    assert not has_meta_parameters(None)


def test_reset_dia_parameters_restores_the_identity_property():
    """HF re-inits modules missing from a checkpoint; reset must undo that."""
    torch.manual_seed(0)
    fusion = EvidenceGuidedFusion(prompt_dim=16, evidence_dim=16, hidden_dim=16)
    adapter = ConceptToEvidenceAdapter(llm_dim=16, visual_dim=16, embed_dim=16, num_heads=2)
    prompt = torch.randn(3, 16)
    for module in (fusion, adapter):  # simulate normal_(0, initializer_range)
        for param in module.parameters():
            with torch.no_grad():
                param.normal_(0, 0.02)

    assert not torch.allclose(fusion(prompt, torch.randn(3, 16))[0], prompt, atol=1e-6)
    fusion.reset_dia_parameters()
    adapter.reset_dia_parameters()
    assert torch.allclose(fusion(prompt, torch.randn(3, 16))[0], prompt, atol=1e-6)
    evidence, attn, _ = adapter(torch.randn(3, 16), torch.randn(3, 16, 4, 4))
    assert torch.isfinite(evidence).all()
    assert torch.allclose(attn.flatten(1).sum(-1), torch.ones(3), atol=1e-5)


def test_fusion_residual_cap():
    torch.manual_seed(0)
    fusion = EvidenceGuidedFusion(
        prompt_dim=16, evidence_dim=16, hidden_dim=16, max_delta_ratio=0.1
    )
    with torch.no_grad():  # break the zero-init to produce a huge residual
        fusion.delta[-1].weight.normal_(0, 5.0)
        fusion.delta[-1].bias.normal_(0, 5.0)
    prompt = torch.randn(7, 16)
    z, stats = fusion(prompt, torch.randn(7, 16))
    ratio = (z - prompt).norm(dim=-1) / prompt.norm(dim=-1)
    assert float(ratio.max()) <= 0.1 + 1e-5
    assert float(stats["delta_ratio"]) <= 0.1 + 1e-5


def test_fusion_uses_the_evidence():
    """Different evidence must give different prompts once delta is non-zero."""
    torch.manual_seed(0)
    fusion = EvidenceGuidedFusion(prompt_dim=16, evidence_dim=16, hidden_dim=16)
    with torch.no_grad():
        fusion.delta[-1].weight.normal_(0, 0.5)
    prompt = torch.randn(4, 16)
    z_a, _ = fusion(prompt, torch.randn(4, 16))
    z_b, _ = fusion(prompt, torch.randn(4, 16))
    assert not torch.allclose(z_a, z_b, atol=1e-4)


# --------------------------------------------------------------------------- #
# attention alignment loss
# --------------------------------------------------------------------------- #
def _attention_on(height, width, region):
    attn = torch.full((1, height, width), 1e-4)
    top, bottom, left, right = region
    attn[0, top:bottom, left:right] = 1.0
    return attn / attn.sum()


def test_alignment_loss_rewards_attention_inside_the_target():
    gt = torch.zeros(1, 64, 64)
    gt[0, 8:16, 8:16] = 1.0
    good = _attention_on(16, 16, (2, 4, 2, 4))   # inside the target
    bad = _attention_on(16, 16, (12, 14, 12, 14))  # far away
    loss_good, n_good = attention_alignment_loss(good, gt)
    loss_bad, n_bad = attention_alignment_loss(bad, gt)
    assert n_good == n_bad == 1
    assert float(loss_good) < float(loss_bad)

    mass_good, _ = attention_mass_in_mask(good, gt)
    mass_bad, _ = attention_mass_in_mask(bad, gt)
    assert float(mass_good) > 0.9 > float(mass_bad)


def test_alignment_loss_kl_mode():
    gt = torch.zeros(2, 32, 32)
    gt[:, 0:16, 0:16] = 1.0
    good = _attention_on(16, 16, (0, 8, 0, 8)).repeat(2, 1, 1)
    bad = _attention_on(16, 16, (8, 16, 8, 16)).repeat(2, 1, 1)
    loss_good, n = attention_alignment_loss(good, gt, mode="kl")
    loss_bad, _ = attention_alignment_loss(bad, gt, mode="kl")
    assert n == 2 and float(loss_good) < float(loss_bad)


def test_alignment_loss_skips_empty_masks():
    """Negative referring samples (no object) must not poison the loss."""
    attn = torch.rand(2, 8, 8)
    attn = attn / attn.flatten(1).sum(-1)[:, None, None]
    gt = torch.zeros(2, 64, 64)
    loss, n_valid = attention_alignment_loss(attn, gt)
    assert n_valid == 0 and float(loss) == 0.0 and torch.isfinite(loss)

    gt[1, 4:12, 4:12] = 1.0
    loss, n_valid = attention_alignment_loss(attn, gt)
    assert n_valid == 1 and torch.isfinite(loss)


def test_alignment_loss_keeps_tiny_objects():
    """A 4x4 target in a 1024x1024 image survives the 64x64 pooling."""
    gt = torch.zeros(1, 1024, 1024)
    gt[0, 512:516, 700:704] = 1.0  # 16 pixels out of 1M -> cell (32, 43) at 64x64
    attn = torch.full((1, 64, 64), 1e-6)
    attn[0, 32, 43] = 1.0
    attn = attn / attn.sum()
    loss, n_valid = attention_alignment_loss(attn, gt)
    assert n_valid == 1 and float(loss) < 0.05

    off_by_one = torch.full((1, 64, 64), 1e-6)
    off_by_one[0, 32, 45] = 1.0
    off_by_one = off_by_one / off_by_one.sum()
    loss_off, _ = attention_alignment_loss(off_by_one, gt)
    assert float(loss_off) > float(loss)  # the loss really localises


# --------------------------------------------------------------------------- #
# end-to-end (module level) gradient flow
# --------------------------------------------------------------------------- #
def test_gradients_reach_adapter_and_fusion():
    torch.manual_seed(0)
    adapter = ConceptToEvidenceAdapter(llm_dim=32, visual_dim=16, embed_dim=16, num_heads=2)
    fusion = EvidenceGuidedFusion(prompt_dim=16, evidence_dim=16, hidden_dim=16)

    concept = torch.randn(2, 32, requires_grad=True)
    features = torch.randn(2, 16, 8, 8)
    prompt = torch.randn(2, 16, requires_grad=True)
    gt = torch.zeros(2, 64, 64)
    gt[:, 8:24, 8:24] = 1.0

    evidence, attn, _ = adapter(concept, features)
    z, _ = fusion(prompt, evidence)
    attn_loss, n_valid = attention_alignment_loss(attn, gt)
    loss = z.square().mean() + attn_loss
    loss.backward()

    assert n_valid == 2
    assert adapter.q_proj.weight.grad is not None
    assert adapter.q_proj.weight.grad.abs().sum() > 0
    assert adapter.k_proj.weight.grad.abs().sum() > 0  # only the attention loss
    assert fusion.gate[-1].weight.grad is not None
    assert concept.grad.abs().sum() > 0                # back into the LLM


def test_dtype_is_preserved_under_bf16_style_inputs():
    adapter = ConceptToEvidenceAdapter(llm_dim=32, visual_dim=16, embed_dim=16, num_heads=2)
    fusion = EvidenceGuidedFusion(prompt_dim=16, evidence_dim=16, hidden_dim=16)
    adapter.to(torch.bfloat16)
    fusion.to(torch.bfloat16)

    evidence, attn, _ = adapter(
        torch.randn(2, 32, dtype=torch.bfloat16), torch.randn(2, 16, 8, 8, dtype=torch.bfloat16)
    )
    z, _ = fusion(torch.randn(2, 16, dtype=torch.bfloat16), evidence)
    assert z.dtype == torch.bfloat16
    assert attn.dtype == torch.float32  # softmax kept in fp32 on purpose
    assert torch.isfinite(z).all()


# --------------------------------------------------------------------------- #
# conversation rewriting
# --------------------------------------------------------------------------- #
def test_insert_con_tokens_clause_style():
    text = "ASSISTANT: Sure, it is [SEG]."
    out = insert_con_tokens(text)
    assert out == "ASSISTANT: Sure, it is [CON], so the segmentation result is [SEG]."
    assert count_tokens(out) == (1, 1)
    assert out.index("[CON]") < out.index("[SEG]")


def test_insert_con_tokens_is_idempotent_and_multi_target():
    text = "It is [SEG] and [SEG]."
    once = insert_con_tokens(text)
    assert count_tokens(once) == (2, 2)
    assert insert_con_tokens(once) == once
    assert insert_con_tokens("no segmentation here") == "no segmentation here"


def test_insert_con_tokens_adjacent_style():
    assert insert_con_tokens("is [SEG].", style="adjacent") == "is [CON] [SEG]."


def test_insert_con_tokens_can_be_disabled():
    """--con_style none is the ablation: no [CON], model falls back to [SEG]."""
    for style in ("none", "", None):
        assert insert_con_tokens("is [SEG].", style=style) == "is [SEG]."


def test_validate_con_seg_pairs():
    assert validate_con_seg_pairs(["a [CON], so the segmentation result is [SEG]."])
    assert validate_con_seg_pairs(["plain text without tokens"])
    assert not validate_con_seg_pairs(["two [SEG] and [SEG] with one [CON]"])


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
