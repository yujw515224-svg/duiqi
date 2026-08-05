import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.DIA_LISAt import (  # noqa: E402
    ContextEvidenceAdapter,
    EvidenceGuideFusion,
    attention_alignment_loss,
)
from model.LISAT import compute_dia_loss_components  # noqa: E402


def test_fusion_zero_init_matches_original_seg():
    torch.manual_seed(0)
    fusion = EvidenceGuideFusion(dim=8, dropout=0.0).eval()
    seg = torch.randn(3, 8)
    con = torch.randn(3, 8)
    evidence = torch.randn(3, 1, 8)

    with torch.no_grad():
        output = fusion(seg, con, evidence)

    assert output.shape == (3, 1, 8)
    assert fusion.res_scale.item() == 0.0
    assert torch.equal(output[:, 0], seg)


def test_context_adapter_k1_attention_sums_to_one():
    torch.manual_seed(1)
    adapter = ContextEvidenceAdapter(
        dim=8,
        num_heads=2,
        num_evidence_tokens=1,
        dropout=0.0,
    ).eval()
    con = torch.randn(4, 8)
    image_embeddings = torch.randn(1, 8, 4, 4)
    image_pe = torch.randn(1, 8, 4, 4)

    evidence, attn_maps = adapter(con, image_embeddings, image_pe=image_pe)

    assert adapter.query_offsets.shape == (1, 8)
    assert adapter.cross_attn.dropout == 0.0
    assert evidence.shape == (4, 1, 8)
    assert attn_maps.shape == (4, 1, 4, 4)
    spatial_sum = attn_maps.flatten(-2).sum(-1)
    assert torch.allclose(spatial_sum, torch.ones_like(spatial_sum), atol=1e-5)


def test_dia_adapter_fusion_forward_backward_smoke():
    torch.manual_seed(3)
    adapter = ContextEvidenceAdapter(
        dim=8,
        num_heads=2,
        num_evidence_tokens=1,
        dropout=0.0,
    )
    fusion = EvidenceGuideFusion(dim=8, dropout=0.0)

    con = torch.randn(2, 8, requires_grad=True)
    seg = torch.randn(2, 8, requires_grad=True)
    image_embeddings = torch.randn(1, 8, 4, 4, requires_grad=True)
    image_pe = torch.randn(1, 8, 4, 4)

    evidence, attn_maps = adapter(con, image_embeddings, image_pe=image_pe)
    fused = fusion(seg, con, evidence)
    loss = fused.square().mean() + attn_maps.square().mean()
    loss.backward()

    assert fused.shape == (2, 1, 8)
    assert torch.isfinite(con.grad).all()
    assert torch.isfinite(seg.grad).all()
    assert torch.isfinite(image_embeddings.grad).all()
    assert fusion.res_scale.grad is not None


def test_attention_alignment_loss_area_kl_cases():
    gt = torch.zeros(1, 8, 8)
    gt[:, 1:3, 1:3] = 1.0

    downsampled = F.interpolate(
        gt.unsqueeze(1),
        size=(4, 4),
        mode="area",
    ).squeeze(1)
    assert downsampled.sum() > 0

    target = downsampled.flatten(1)
    target = target / target.sum(dim=-1, keepdim=True)
    correct_attn = target.view(1, 1, 4, 4).clone().requires_grad_(True)
    correct_loss = attention_alignment_loss(correct_attn, gt)
    assert correct_loss.item() < 1e-4
    correct_loss.backward()
    assert correct_attn.grad is not None
    assert torch.isfinite(correct_attn.grad).all()

    wrong_target = torch.roll(target, shifts=1, dims=-1)
    wrong_attn = wrong_target.view(1, 1, 4, 4).clone().requires_grad_(True)
    wrong_loss = attention_alignment_loss(wrong_attn, gt)
    assert wrong_loss.item() > correct_loss.item()
    wrong_loss.backward()
    assert torch.isfinite(wrong_attn.grad).all()

    empty_gt = torch.zeros(2, 8, 8)
    empty_attn = torch.full((2, 1, 4, 4), 1.0 / 16.0, requires_grad=True)
    empty_loss = attention_alignment_loss(empty_attn, empty_gt)
    assert empty_loss.item() == 0.0
    empty_loss.backward()
    assert torch.isfinite(empty_attn.grad).all()


def test_all_negative_loss_backward_is_ce_only():
    ce_loss = torch.tensor(2.0, requires_grad=True)
    pred_masks = [torch.empty(0, 4, 4)]
    gt_masks = [torch.empty(0, 4, 4)]
    loss_dict = compute_dia_loss_components(
        ce_loss=ce_loss,
        pred_masks=pred_masks,
        gt_masks=gt_masks,
        attn_maps_list=[None],
        bce_loss_weight=2.0,
        dice_loss_weight=0.5,
        attn_loss_weight=0.02,
    )

    assert torch.allclose(loss_dict["loss"], ce_loss)
    assert loss_dict["mask_bce_loss"].item() == 0.0
    assert loss_dict["mask_dice_loss"].item() == 0.0
    assert loss_dict["attn_alignment_loss"].item() == 0.0
    assert loss_dict["num_positive_masks"].item() == 0.0
    assert loss_dict["num_valid_attn_masks"].item() == 0.0
    loss_dict["loss"].backward()
    assert ce_loss.grad.item() == 1.0


def test_mixed_positive_negative_loss_backward_is_finite():
    torch.manual_seed(2)
    ce_loss = torch.tensor(1.0, requires_grad=True)
    pred_pos = torch.randn(1, 4, 4, requires_grad=True)
    gt_pos = torch.zeros(1, 4, 4)
    gt_pos[:, 1:3, 1:3] = 1.0
    pred_neg = torch.empty(0, 4, 4)
    gt_neg = torch.empty(0, 4, 4)
    attn = torch.rand(1, 1, 2, 2)
    attn = attn / attn.flatten(-2).sum(-1).view(1, 1, 1, 1)
    attn = attn.detach().requires_grad_(True)

    loss_dict = compute_dia_loss_components(
        ce_loss=ce_loss,
        pred_masks=[pred_pos, pred_neg],
        gt_masks=[gt_pos, gt_neg],
        attn_maps_list=[attn, None],
        bce_loss_weight=2.0,
        dice_loss_weight=0.5,
        attn_loss_weight=0.02,
    )

    assert loss_dict["num_positive_masks"].item() == 1.0
    assert loss_dict["num_valid_attn_masks"].item() == 1.0
    assert torch.isfinite(loss_dict["loss"])
    loss_dict["loss"].backward()
    assert torch.isfinite(pred_pos.grad).all()
    assert torch.isfinite(attn.grad).all()
    assert ce_loss.grad.item() == 1.0


def test_collate_accepts_mixed_vqa_and_segmentation_protocols():
    from dataloaders import trainval_dataset as tvd

    class FakeTokenizer:
        pad_token_id = 0
        model_max_length = 64

    def fake_tokenize(conversation_list, tokenizer, padding="right"):
        input_ids = torch.ones(len(conversation_list), 6, dtype=torch.long)
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
        vqa_sample = (
            "vqa.jpg",
            image,
            image_clip,
            ["USER: <image>\nASSISTANT: answer"],
            torch.empty(0, 8, 8),
            [(8, 8), (8, 8)],
            [False],
            None,
            None,
        )
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
        output = tvd.collate_fn_train(
            [vqa_sample, seg_sample],
            tokenizer=FakeTokenizer(),
            conv_type="llava_v1",
        )
    finally:
        tvd.tokenize_and_pad = old_tokenize
        tvd.handle_conversation_specifics = old_targets

    assert output["images"].shape == (2, 3, 8, 8)
    assert output["images_clip"].shape == (2, 3, 4, 4)
    assert len(output["masks_list"]) == 2
    assert output["masks_list"][0].shape[0] == 0
    assert output["masks_list"][1].shape[0] == 1
    assert output["offset"].tolist() == [0, 1, 2]


def test_auto_resume_argparse_defaults():
    from train_lisat import parse_args

    assert parse_args([]).auto_resume is False
    assert parse_args(["--auto_resume"]).auto_resume is True
    assert parse_args(["--no_auto_resume"]).auto_resume is False
    args = parse_args([])
    assert args.use_dia is False
    assert parse_args(["--use_dia"]).use_dia is True
    assert args.dia_num_evidence_tokens == 1
    assert args.dia_num_heads == 8
    assert args.dia_attn_dropout == 0.0
    assert math.isclose(args.attn_loss_weight, 0.02)


if __name__ == "__main__":
    test_fusion_zero_init_matches_original_seg()
    test_context_adapter_k1_attention_sums_to_one()
    test_dia_adapter_fusion_forward_backward_smoke()
    test_attention_alignment_loss_area_kl_cases()
    test_all_negative_loss_backward_is_ce_only()
    test_mixed_positive_negative_loss_backward_is_finite()
    test_collate_accepts_mixed_vqa_and_segmentation_protocols()
    test_auto_resume_argparse_defaults()
    print("DIA pure-structure tests passed.")
