# -*- coding: utf-8 -*-
"""CPU tests for the batch bookkeeping of DIA-LISAt.

These cover the parts that are easy to get wrong once several images, several
conversations per image and several targets per conversation are mixed inside a
single batch:

* which image does a given ``[SEG]`` attend to,
* which concept token feeds it,
* how the flat ``[n_seg, ...]`` rows are regrouped per image for the decoder.

    python tests/test_dia_batching.py
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dia_modules import (  # noqa: E402
    ConceptToEvidenceAdapter,
    EvidenceGuidedFusion,
    build_special_token_mask,
    compute_dia_prompts,
    decode_masks_with_sam,
    rows_to_image_index,
    split_by_token_offset,
)

SEG_ID, CON_ID, NUM_PATCHES = 32000, 32001, 4  # tiny "image" to keep tests fast


def _batch(rows, seq_len=16):
    """Build input_ids where rows is a list of (con_pos, seg_pos) lists."""
    input_ids = torch.zeros(len(rows), seq_len, dtype=torch.long)
    for r, pairs in enumerate(rows):
        for con_pos, seg_pos in pairs:
            if con_pos is not None:
                input_ids[r, con_pos] = CON_ID
            input_ids[r, seg_pos] = SEG_ID
    seg_mask = build_special_token_mask(input_ids, SEG_ID, NUM_PATCHES)
    con_mask = build_special_token_mask(input_ids, CON_ID, NUM_PATCHES)
    return input_ids, seg_mask, con_mask


# --------------------------------------------------------------------------- #
def test_rows_to_image_index():
    offset = torch.tensor([0, 2, 3, 6])  # 3 images, 6 conversation rows
    assert rows_to_image_index(6, offset).tolist() == [0, 0, 1, 2, 2, 2]


def test_split_by_token_offset_groups_targets_per_image():
    _, seg_mask, _ = _batch([[(4, 6)], [(4, 6), (8, 10)], [(4, 6)]])
    offset = torch.tensor([0, 1, 3])  # image0 -> row0, image1 -> rows 1,2
    values = torch.arange(4).float().unsqueeze(-1)  # 1 + 2 + 1 = 4 [SEG]
    chunks = split_by_token_offset(values, seg_mask, offset)
    assert [c.shape[0] for c in chunks] == [1, 3]
    assert chunks[0].flatten().tolist() == [0.0]
    assert chunks[1].flatten().tolist() == [1.0, 2.0, 3.0]


def test_split_handles_rows_without_targets():
    _, seg_mask, _ = _batch([[], [(4, 6)]])
    offset = torch.tensor([0, 1, 2])
    chunks = split_by_token_offset(torch.ones(1, 3), seg_mask, offset)
    assert chunks[0].shape[0] == 0 and chunks[1].shape[0] == 1


# --------------------------------------------------------------------------- #
def _dia_parts(llm_dim=32, visual_dim=16, seed=0):
    torch.manual_seed(seed)
    text_fc = nn.Linear(llm_dim, visual_dim)
    adapter = ConceptToEvidenceAdapter(
        llm_dim=llm_dim, visual_dim=visual_dim, embed_dim=visual_dim, num_heads=1
    )
    fusion = EvidenceGuidedFusion(
        prompt_dim=visual_dim, evidence_dim=visual_dim, hidden_dim=visual_dim
    )
    return text_fc, adapter, fusion


def test_compute_dia_prompts_shapes_and_order():
    llm_dim, visual_dim = 32, 16
    input_ids, seg_mask, con_mask = _batch([[(4, 6)], [(4, 6), (8, 10)]])
    hidden = torch.randn(2, seg_mask.shape[1], llm_dim)
    image_embeddings = torch.randn(2, visual_dim, NUM_PATCHES, NUM_PATCHES)
    row_to_image = rows_to_image_index(2, torch.tensor([0, 1, 2]))
    text_fc, adapter, fusion = _dia_parts(llm_dim, visual_dim)

    z, attn, stats = compute_dia_prompts(
        hidden, image_embeddings, seg_mask, con_mask, row_to_image,
        text_fc, adapter, fusion,
    )
    assert z.shape == (3, visual_dim)          # 1 + 2 targets, row-major order
    assert attn.shape == (3, NUM_PATCHES, NUM_PATCHES)
    assert float(stats["con_hit_rate"]) == 1.0
    # zero-init fusion => z is exactly LISAt's projected [SEG] embedding
    expected = text_fc(hidden.reshape(-1, llm_dim)[seg_mask.reshape(-1)])
    assert torch.allclose(z, expected, atol=1e-6)


def test_each_seg_attends_to_its_own_image():
    """A [SEG] of image 1 must never look at the features of image 0."""
    llm_dim = visual_dim = 32
    input_ids, seg_mask, con_mask = _batch([[(4, 6)], [(4, 6)]])
    text_fc, adapter, fusion = _dia_parts(llm_dim, visual_dim)
    with torch.no_grad():  # identity projections -> attention = feature match
        adapter.q_proj.weight.copy_(torch.eye(visual_dim))
        adapter.k_proj.weight.copy_(torch.eye(visual_dim))
        adapter.q_proj.bias.zero_()
        adapter.k_proj.bias.zero_()

    image_embeddings = torch.randn(2, visual_dim, NUM_PATCHES, NUM_PATCHES)
    hidden = torch.zeros(2, seg_mask.shape[1], llm_dim)
    # row 0 asks for patch (0,1) of image 0, row 1 asks for patch (3,2) of image 1
    con_positions = torch.nonzero(con_mask, as_tuple=False)
    hidden[0, con_positions[0, 1]] = image_embeddings[0, :, 0, 1]
    hidden[1, con_positions[1, 1]] = image_embeddings[1, :, 3, 2]

    row_to_image = rows_to_image_index(2, torch.tensor([0, 1, 2]))
    _, attn, _ = compute_dia_prompts(
        hidden, image_embeddings, seg_mask, con_mask, row_to_image,
        text_fc, adapter, fusion,
    )
    assert attn[0].flatten().argmax().item() == 0 * NUM_PATCHES + 1
    assert attn[1].flatten().argmax().item() == 3 * NUM_PATCHES + 2


def test_two_conversations_of_the_same_image_share_its_features():
    llm_dim = visual_dim = 32
    _, seg_mask, con_mask = _batch([[(4, 6)], [(4, 6)]])
    text_fc, adapter, fusion = _dia_parts(llm_dim, visual_dim)
    with torch.no_grad():
        adapter.q_proj.weight.copy_(torch.eye(visual_dim))
        adapter.k_proj.weight.copy_(torch.eye(visual_dim))
        adapter.q_proj.bias.zero_()
        adapter.k_proj.bias.zero_()

    image_embeddings = torch.randn(1, visual_dim, NUM_PATCHES, NUM_PATCHES)
    hidden = torch.zeros(2, seg_mask.shape[1], llm_dim)
    con_positions = torch.nonzero(con_mask, as_tuple=False)
    hidden[0, con_positions[0, 1]] = image_embeddings[0, :, 2, 2]
    hidden[1, con_positions[1, 1]] = image_embeddings[0, :, 1, 0]

    row_to_image = rows_to_image_index(2, torch.tensor([0, 2]))  # both rows -> image 0
    _, attn, _ = compute_dia_prompts(
        hidden, image_embeddings, seg_mask, con_mask, row_to_image,
        text_fc, adapter, fusion,
    )
    assert attn[0].flatten().argmax().item() == 2 * NUM_PATCHES + 2
    assert attn[1].flatten().argmax().item() == 1 * NUM_PATCHES + 0


def test_missing_concept_token_still_produces_a_prompt():
    llm_dim, visual_dim = 32, 16
    _, seg_mask, con_mask = _batch([[(None, 6)]])
    hidden = torch.randn(1, seg_mask.shape[1], llm_dim)
    text_fc, adapter, fusion = _dia_parts(llm_dim, visual_dim)
    z, attn, stats = compute_dia_prompts(
        hidden,
        torch.randn(1, visual_dim, NUM_PATCHES, NUM_PATCHES),
        seg_mask,
        con_mask,
        rows_to_image_index(1, torch.tensor([0, 1])),
        text_fc,
        adapter,
        fusion,
    )
    assert z.shape == (1, visual_dim) and torch.isfinite(z).all()
    assert float(stats["con_hit_rate"]) == 0.0  # fell back to the [SEG] state


# --------------------------------------------------------------------------- #
class _FakePromptEncoder(nn.Module):
    def __init__(self, dim=16, size=4):
        super().__init__()
        self.dim, self.size = dim, size
        self.calls = []

    def forward(self, points=None, boxes=None, masks=None, text_embeds=None):
        self.calls.append(text_embeds.shape)
        n = text_embeds.shape[0]
        return (
            torch.zeros(n, 1, self.dim),
            torch.zeros(n, self.dim, self.size, self.size),
        )

    def get_dense_pe(self):
        return torch.zeros(1, self.dim, self.size, self.size)


class _FakeVisualModel(nn.Module):
    def __init__(self, dim=16, size=4):
        super().__init__()
        self.prompt_encoder = _FakePromptEncoder(dim, size)

    def mask_decoder(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                     dense_prompt_embeddings, multimask_output):
        n = sparse_prompt_embeddings.shape[0]
        return torch.zeros(n, 1, 8, 8), torch.zeros(n, 1)

    @staticmethod
    def postprocess_masks(low_res_masks, input_size, original_size):
        n = low_res_masks.shape[0]
        return torch.zeros(n, 1, *original_size)


def test_decode_masks_with_sam_handles_counts_and_none():
    visual_model = _FakeVisualModel()
    image_embeddings = torch.zeros(2, 16, 4, 4)
    shapes = [((32, 32), (40, 50)), ((32, 32), (60, 70))]

    prompts = [torch.randn(2, 16), None]
    masks = decode_masks_with_sam(visual_model, prompts, image_embeddings, shapes)
    assert masks[0].shape == (2, 40, 50)   # two targets for image 0
    assert masks[1].shape == (60, 70)      # placeholder for "no [SEG]"
    assert visual_model.prompt_encoder.calls == [torch.Size([2, 1, 16])]


def test_decode_masks_skips_degenerate_sizes():
    visual_model = _FakeVisualModel()
    masks = decode_masks_with_sam(
        visual_model,
        [torch.randn(1, 16)],
        torch.zeros(1, 16, 4, 4),
        [((32, 32), (0, 0))],
    )
    assert masks[0].shape == (0, 32, 32)


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
