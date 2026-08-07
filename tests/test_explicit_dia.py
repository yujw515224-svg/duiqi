import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders import trainval_dataset as tvd  # noqa: E402
from model.LISAT import LISATForCausalLM  # noqa: E402


class _Encoded:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeTokenizer:
    pad_token_id = 0
    model_max_length = 512

    def __call__(self, text, add_special_tokens=False):
        if text == "[CON]":
            return _Encoded([10])
        if text == "[SEG]":
            return _Encoded([20])
        if text == "[CON][SEG]":
            return _Encoded([10, 20])
        raise AssertionError(f"Unexpected direct tokenization request: {text}")


def _assert_raises(fn, exc_type=RuntimeError):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__}.")


def test_insert_explicit_conversation_pairs():
    assert tvd.insert_explicit_con("ASSISTANT: [SEG]") == "ASSISTANT: [CON][SEG]"
    assert tvd.insert_explicit_con("ASSISTANT: text only") == "ASSISTANT: text only"
    assert (
        tvd.insert_explicit_con("ASSISTANT: [CON][SEG]")
        == "ASSISTANT: [CON][SEG]"
    )
    _assert_raises(lambda: tvd.insert_explicit_con("ASSISTANT: [CON] [SEG]"))
    _assert_raises(lambda: tvd.insert_explicit_con("ASSISTANT: [CON][SEG] [SEG]"))


def test_validate_explicit_pairs_accepts_only_adjacent_pairs():
    tokenizer = FakeTokenizer()
    good = torch.tensor([[1, 10, 20, 2, 0], [1, 2, 0, 0, 0]])
    mask = good.ne(tokenizer.pad_token_id)
    tvd.validate_explicit_pairs(good, mask, tokenizer)

    bad_missing_con = torch.tensor([[1, 20, 2]])
    _assert_raises(
        lambda: tvd.validate_explicit_pairs(
            bad_missing_con,
            bad_missing_con.ne(tokenizer.pad_token_id),
            tokenizer,
        )
    )

    bad_non_adjacent = torch.tensor([[1, 10, 99, 20]])
    _assert_raises(
        lambda: tvd.validate_explicit_pairs(
            bad_non_adjacent,
            bad_non_adjacent.ne(tokenizer.pad_token_id),
            tokenizer,
        )
    )

    bad_reversed = torch.tensor([[1, 20, 10, 2]])
    _assert_raises(
        lambda: tvd.validate_explicit_pairs(
            bad_reversed,
            bad_reversed.ne(tokenizer.pad_token_id),
            tokenizer,
        )
    )


def test_collate_train_inserts_and_validates_explicit_pair():
    def fake_tokenize(conversation_list, tokenizer, padding="right"):
        rows = []
        for conversation in conversation_list:
            if "[CON][SEG]" in conversation:
                rows.append([1, 10, 20, 2])
            elif "[SEG]" in conversation:
                rows.append([1, 20, 2, 0])
            else:
                rows.append([1, 2, 0, 0])
        input_ids = torch.tensor(rows, dtype=torch.long)
        return input_ids, input_ids.ne(tokenizer.pad_token_id)

    def fake_targets(input_ids, conversation_list, tokenizer, conv_type):
        return input_ids.clone()

    old_tokenize = tvd.tokenize_and_pad
    old_targets = tvd.handle_conversation_specifics
    tvd.tokenize_and_pad = fake_tokenize
    tvd.handle_conversation_specifics = fake_targets
    try:
        image = torch.zeros(3, 8, 8)
        image_clip = torch.zeros(3, 4, 4)
        seg_sample = (
            "seg.jpg",
            image,
            image_clip,
            ["USER: <image>\nASSISTANT: [SEG]"],
            torch.ones(1, 8, 8),
            [(8, 8), (8, 8)],
            [True],
            [1],
            [0],
        )
        vqa_sample = (
            "vqa.jpg",
            image,
            image_clip,
            ["USER: <image>\nASSISTANT: text only"],
            torch.empty(0, 8, 8),
            [(8, 8), (8, 8)],
            [False],
            None,
            None,
        )
        output = tvd.collate_fn_train(
            [seg_sample, vqa_sample],
            tokenizer=FakeTokenizer(),
            conv_type="llava_v1",
            explicit_con_in_conversation=True,
        )
    finally:
        tvd.tokenize_and_pad = old_tokenize
        tvd.handle_conversation_specifics = old_targets

    assert "[CON][SEG]" in output["conversation_list"][0]
    assert output["input_ids"][0, 1].item() == 10
    assert output["input_ids"][0, 2].item() == 20
    assert output["offset"].tolist() == [0, 1, 2]


def _fake_model(explicit=True):
    model = object.__new__(LISATForCausalLM)
    model.seg_token_idx = 20
    model.con_token_idx = 10
    model.use_dia = True
    model.explicit_con_in_conversation = explicit
    model.dia_fusion_mode = "legacy"

    class FakeVisionTower:
        num_patches = 3

    model.get_vision_tower = lambda: FakeVisionTower()
    return model


def test_build_dia_token_masks_splits_adjacent_hidden_states():
    model = _fake_model(explicit=True)
    input_ids = torch.tensor([[1, 10, 20, 2, 0], [1, 2, 0, 0, 0]])
    seg_mask, con_mask = model.build_dia_token_masks(input_ids, hidden_len=8)

    assert seg_mask.shape == con_mask.shape == (2, 8)
    assert seg_mask[0].sum().item() == 1
    assert con_mask[0].sum().item() == 1
    assert not torch.equal(seg_mask, con_mask)
    assert con_mask[0].nonzero().item() + 1 == seg_mask[0].nonzero().item()
    assert seg_mask[1].sum().item() == 0
    assert con_mask[1].sum().item() == 0

    bad = torch.tensor([[1, 10, 99, 20]])
    _assert_raises(lambda: model.build_dia_token_masks(bad, hidden_len=8))


def test_structural_dia_keeps_shared_mask_fallback():
    model = _fake_model(explicit=False)
    input_ids = torch.tensor([[1, 20, 2, 0]])
    seg_mask, con_mask = model.build_dia_token_masks(input_ids, hidden_len=8)
    assert torch.equal(seg_mask, con_mask)


def test_explicit_generate_pred_masks_rejects_mismatched_prompts_before_sam():
    model = _fake_model(explicit=True)
    seg_embeddings = [torch.randn(2, 8)]
    con_embeddings = [torch.randn(1, 8)]
    image_embeddings = torch.randn(1, 8, 4, 4)
    sam_mask_shape_list = [((8, 8), (8, 8))]

    _assert_raises(
        lambda: LISATForCausalLM.generate_pred_masks(
            model,
            seg_embeddings,
            con_embeddings,
            image_embeddings,
            sam_mask_shape_list,
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
