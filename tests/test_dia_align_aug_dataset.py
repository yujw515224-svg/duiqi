import torch

from dataloaders.dia_align_aug_dataset import (
    DIAAlignAugDataset,
    build_density_matched_description,
    infer_phrase_from_conversation,
    make_density_reduced_view,
    make_target_removed_view,
)


class _DummySource:
    def __init__(self):
        self.image = torch.arange(3 * 8 * 8, dtype=torch.float32).view(3, 8, 8)
        self.image_clip = torch.arange(3 * 4 * 4, dtype=torch.float32).view(3, 4, 4)
        self.mask = torch.zeros(8, 8)
        self.mask[2:6, 3:7] = 1.0

    def __getitem__(self, idx):
        return (
            "dummy.png",
            self.image.clone(),
            self.image_clip.clone(),
            ["USER: <image>\nPlease segment solar panel###ASSISTANT: [SEG]</s>"],
            self.mask[None].clone(),
            [(8, 8), (8, 8)],
            [True],
            ["dummy_ref"],
            [7],
        )


def test_density_reduced_view_suppresses_background_and_keeps_target():
    image = torch.ones(3, 4, 4)
    image[:, :2, :2] = 10.0
    mask = torch.zeros(4, 4)
    mask[:2, :2] = 1.0

    edited = make_density_reduced_view(image, mask, background_scale=0.2)

    assert torch.equal(edited[:, :2, :2], image[:, :2, :2])
    assert torch.allclose(edited[:, 2:, 2:], image[:, 2:, 2:] * 0.2)


def test_target_removed_view_replaces_target_with_background_mean():
    image = torch.zeros(3, 4, 4)
    image[:, :2, :2] = 10.0
    image[:, 2:, 2:] = 2.0
    mask = torch.zeros(4, 4)
    mask[:2, :2] = 1.0

    edited = make_target_removed_view(image, mask)

    assert not torch.equal(edited[:, :2, :2], image[:, :2, :2])
    assert torch.allclose(edited[:, 2:, 2:], image[:, 2:, 2:])


def test_density_matched_description_contains_remote_sensing_cues():
    mask = torch.zeros(10, 10)
    mask[:2, :2] = 1.0

    text = build_density_matched_description("airport runway", mask)

    assert "target concept: airport runway" in text
    assert "Remote-sensing evidence" in text
    assert "upper-left" in text
    assert "4.0%" in text


def test_infer_phrase_from_conversation_removes_instruction_words():
    conversation = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "USER: <image>\nPlease identify bus located at top center of image. "
        "Inspect for mentioned area and provide segmentation map if visible. "
        "ASSISTANT: Yes, [SEG].</s>"
    )

    assert infer_phrase_from_conversation(conversation) == "bus top center"


def test_aug_dataset_positive_and_negative_protocols():
    ds = DIAAlignAugDataset.__new__(DIAAlignAugDataset)
    ds.samples_per_epoch = 10
    ds.source_datasets = [_DummySource()]
    ds.source_sample_rate = [1.0]
    ds.positive_prob = 1.0
    ds.background_scale = 0.2
    ds.mask_dilation = 0
    ds.question_list = ["<image>\nLocate: {description}"]
    ds.answer_list = ["Yes, {class_name}: [SEG]."]
    ds.neg_answer_list = ["No {class_name} is present."]

    pos = ds[0]
    assert pos[4].shape == (1, 8, 8)
    assert pos[6] == [True]
    assert "[SEG]" in pos[3][0]
    assert "Remote-sensing evidence" in pos[3][0]

    ds.positive_prob = 0.0
    neg = ds[0]
    assert neg[4].shape == (0, 8, 8)
    assert neg[6] == [False]
    assert "[SEG]" not in neg[3][0]
    assert "[CON]" in neg[3][0]
    assert "Remote-sensing evidence" in neg[3][0]
    assert "upper-left" not in neg[3][0]
    assert "%" not in neg[3][0]
