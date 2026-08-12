import random
import re

import numpy as np
import torch
import torch.nn.functional as F

from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN

from .base_dataset import BaseDataset
from .qa_template import NEG_ANSWER_TEMPLATE, SHORT_ANSWER_TEMPLATE


DIA_ALIGN_AUG_QUESTION_TEMPLATE = [
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + (
        "The image has been edited to reduce unrelated visual evidence. "
        "Use the detailed target description to locate the supported evidence "
        "and segment it only if present: {description}"
    ),
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + (
        "Focus on the target evidence described below. Segment the target if the "
        "edited image still contains it: {description}"
    ),
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + (
        "The scene may hide or remove the requested target. Check the visual "
        "evidence against this description and provide a mask only when it is "
        "present: {description}"
    ),
]


def _mask_to_tensor_size(mask, tensor_shape, sam_input_shape=None, dilation=0):
    """Resize an unpadded GT mask to the spatial size of an image tensor."""
    if mask.ndim != 2:
        raise RuntimeError(f"mask must be [H, W], got {tuple(mask.shape)}")
    h, w = int(tensor_shape[-2]), int(tensor_shape[-1])
    target_h, target_w = sam_input_shape or (h, w)
    resized = F.interpolate(
        mask.float()[None, None],
        size=(int(target_h), int(target_w)),
        mode="nearest",
    )[0, 0]
    padded = resized.new_zeros((h, w))
    copy_h = min(h, resized.shape[0])
    copy_w = min(w, resized.shape[1])
    padded[:copy_h, :copy_w] = resized[:copy_h, :copy_w]
    if dilation > 0:
        kernel = int(dilation) * 2 + 1
        padded = F.max_pool2d(
            padded[None, None],
            kernel_size=kernel,
            stride=1,
            padding=int(dilation),
        )[0, 0]
    return (padded > 0.5).float()


def _background_mean_fill(image, mask):
    background = mask < 0.5
    if background.any():
        fill = image[:, background].mean(dim=1)
    else:
        fill = image.flatten(1).mean(dim=1)
    return fill[:, None, None]


def make_density_reduced_view(image, mask, background_scale=0.20):
    """Keep the target and suppress unrelated regions in normalized image space."""
    mask = mask.to(device=image.device, dtype=image.dtype).clamp(0, 1)
    bg_scale = float(background_scale)
    return image * mask.unsqueeze(0) + image * bg_scale * (1.0 - mask).unsqueeze(0)


def make_target_removed_view(image, mask):
    """Remove the target evidence by filling it with the image background mean."""
    mask = mask.to(device=image.device, dtype=image.dtype).clamp(0, 1)
    fill = _background_mean_fill(image, mask)
    return image * (1.0 - mask).unsqueeze(0) + fill * mask.unsqueeze(0)


def _location_words(mask):
    ys, xs = torch.where(mask > 0.5)
    if ys.numel() == 0:
        return "unknown location", 0.0, "unknown extent"
    h, w = mask.shape
    y0, y1 = ys.min().item(), ys.max().item()
    x0, x1 = xs.min().item(), xs.max().item()
    cy = (y0 + y1) / 2.0 / max(1, h)
    cx = (x0 + x1) / 2.0 / max(1, w)
    vertical = "upper" if cy < 0.33 else "lower" if cy > 0.67 else "central"
    horizontal = "left" if cx < 0.33 else "right" if cx > 0.67 else "middle"
    area = float((mask > 0.5).float().mean().item())
    bw = (x1 - x0 + 1) / max(1, w)
    bh = (y1 - y0 + 1) / max(1, h)
    if bw > bh * 1.8:
        shape = "horizontally elongated"
    elif bh > bw * 1.8:
        shape = "vertically elongated"
    elif area < 0.02:
        shape = "small and compact"
    else:
        shape = "compact"
    return f"{vertical}-{horizontal}", area, shape


def build_density_matched_description(phrase, mask):
    location, area, shape = _location_words(mask)
    phrase = (phrase or "target object").strip().lower()
    return (
        f"target concept: {phrase}. "
        f"Remote-sensing evidence: a {shape} region around the {location} "
        f"part of the image, occupying about {area * 100.0:.1f}% of the scene. "
        "Use local visual evidence rather than global scene context."
    )


def _clean_target_phrase(text):
    text = (text or "").replace(DEFAULT_IMAGE_TOKEN, " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[A-Z]+\]", " ", text)
    text = re.sub(r"[^A-Za-z0-9 _+-]", " ", text)
    stop_words = {
        "a",
        "an",
        "and",
        "any",
        "area",
        "as",
        "at",
        "be",
        "between",
        "can",
        "check",
        "confirm",
        "could",
        "depicted",
        "detect",
        "display",
        "do",
        "does",
        "exist",
        "exists",
        "feature",
        "featured",
        "find",
        "found",
        "for",
        "generate",
        "give",
        "if",
        "image",
        "in",
        "include",
        "indicated",
        "inspect",
        "is",
        "it",
        "its",
        "kindly",
        "locate",
        "located",
        "map",
        "mentioned",
        "of",
        "occurrence",
        "part",
        "please",
        "presence",
        "present",
        "provide",
        "region",
        "review",
        "search",
        "see",
        "segment",
        "segmentation",
        "share",
        "show",
        "specified",
        "supply",
        "target",
        "the",
        "there",
        "this",
        "to",
        "trace",
        "verify",
        "visible",
        "whether",
        "with",
        "would",
        "you",
    }
    tokens = [tok.lower() for tok in text.split()]
    tokens = [tok for tok in tokens if tok not in stop_words]
    phrase = " ".join(tokens).strip()
    return phrase or "target object"


def infer_phrase_from_conversation(conversation):
    text = conversation.replace(DEFAULT_IMAGE_TOKEN, " ")
    text = re.sub(r"A chat between .*?</s>", " ", text)
    text = re.sub(r"^(.*?)USER:\s*", "", text, flags=re.DOTALL)
    text = re.split(r"ASSISTANT:|###\s*ASSISTANT:", text, maxsplit=1)[0]
    text = re.sub(r"###.*", " ", text)

    patterns = [
        r"\bif\s+(?:yes|present|detected|found|it(?:'s| is)?\s+there).*$",
        r"\bif\s+(?:it|this|the)\b.*$",
        r"\bprovide\b.*$",
        r"\bsupply\b.*$",
        r"\bshow\b.*$",
        r"\bdisplay\b.*$",
        r"\bgenerate\b.*$",
        r"\bshare\b.*$",
        r"\boffer\b.*$",
        r"\bpresent\b.*$",
    ]
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.I)

    match_patterns = [
        r"\bfor\s+(?:the\s+presence\s+of\s+)?(.+?)(?:\s+in\s+(?:this|the)\s+image|\s*$)",
        r"\bwhether\s+(.+?)\s+(?:appears|is|exists|visible|present|featured|depicted)\b",
        r"\bif\s+(.+?)\s+(?:appears|is|exists|visible|present|featured|depicted)\b",
        r"\b(?:contains|include|includes|showcase|feature|features)\s+(.+?)(?:\s+in\s+(?:this|the)\s+image|\s*$)",
        r"\b(?:detect|spot|find|identify|locate|search)\s+(.+?)(?:\s+in\s+(?:this|the)\s+image|\s*$)",
        r"\b(?:presence|occurrence|trace|sign)\s+of\s+(.+?)(?:\s+in\s+(?:this|the)\s+image|\s*$)",
    ]
    for pattern in match_patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            phrase = _clean_target_phrase(match.group(1))
            if phrase != "target object":
                return phrase[:64]

    phrase = _clean_target_phrase(text)
    return phrase[:64]


class DIAAlignAugDataset(BaseDataset):
    """On-the-fly edited data for DIA concept-evidence alignment.

    This dataset wraps existing segmentation datasets. It generates:
      * density-reduced positive samples: target kept, unrelated background
        suppressed, detailed description, normal [SEG] answer;
      * target-removed negative samples: target region filled from background,
        same detailed description, negative answer without [SEG].

    The original paper datasets remain untouched; this branch is used only when
    the training dataset list explicitly includes ``dia_align_aug``.
    """

    def __init__(
        self,
        base_image_dir,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        source_dataset="sem_seg||refer_seg||reason_seg||geo_reason_seg",
        source_sample_rate="15,15,1,36",
        sem_seg_data="ade20k||cocostuff||pascal_part||paco_lvis",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        neg_refer_seg_data="R-refcocog||R-refcoco||R-refcoco+",
        correct_refer_seg_data="fprefcocog||fprefcoco||fprefcoco+",
        reason_seg_data="ReasonSeg|train",
        geo_reason_seg_data="GeoReasonSeg|train",
        positive_prob=0.50,
        background_scale=0.20,
        mask_dilation=2,
    ):
        super().__init__(vision_tower, samples_per_epoch, image_size)
        self.base_image_dir = base_image_dir
        self.num_classes_per_sample = num_classes_per_sample
        self.positive_prob = float(positive_prob)
        self.background_scale = float(background_scale)
        self.mask_dilation = int(mask_dilation)
        self.question_list = DIA_ALIGN_AUG_QUESTION_TEMPLATE
        self.answer_list = SHORT_ANSWER_TEMPLATE
        self.neg_answer_list = NEG_ANSWER_TEMPLATE

        self.source_datasets = []
        for name in source_dataset.split("||"):
            if name == "sem_seg":
                from .sem_seg_dataset import SemSegDataset

                self.source_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        vision_tower,
                        samples_per_epoch,
                        image_size,
                        num_classes_per_sample,
                        sem_seg_data,
                    )
                )
            elif name == "refer_seg":
                from .refer_seg_dataset import ReferSegDataset

                self.source_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        vision_tower,
                        samples_per_epoch,
                        image_size,
                        num_classes_per_sample,
                        refer_seg_data,
                    )
                )
            elif name == "neg_refer_seg":
                from .refer_seg_dataset import ReferSegDataset

                self.source_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        vision_tower,
                        samples_per_epoch,
                        image_size,
                        num_classes_per_sample,
                        neg_refer_seg_data,
                    )
                )
            elif name == "correct_refer_seg":
                from .refer_seg_dataset import ReferSegDataset

                self.source_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        vision_tower,
                        samples_per_epoch,
                        image_size,
                        num_classes_per_sample,
                        correct_refer_seg_data,
                    )
                )
            elif name == "reason_seg":
                from .reason_seg_dataset import ReasonSegDataset

                self.source_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        vision_tower,
                        samples_per_epoch,
                        image_size,
                        num_classes_per_sample,
                        reason_seg_data,
                        use_fp=False,
                        is_train=True,
                    )
                )
            elif name == "geo_reason_seg":
                from .reason_seg_dataset import ReasonSegDataset

                self.source_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        vision_tower,
                        samples_per_epoch,
                        image_size,
                        num_classes_per_sample,
                        reason_seg_data=geo_reason_seg_data,
                        use_fp=False,
                        is_train=True,
                    )
                )
            elif name == "refsegrs":
                from .refsegrs_dataset import RefSegRSDataset

                self.source_datasets.append(
                    RefSegRSDataset(
                        base_image_dir,
                        vision_tower,
                        samples_per_epoch,
                        image_size,
                        num_classes_per_sample,
                        split="train",
                        is_train=True,
                    )
                )
            else:
                raise ValueError(f"Unknown DIA alignment source dataset: {name}")

        if not self.source_datasets:
            raise RuntimeError("DIAAlignAugDataset needs at least one source dataset.")
        rates = np.array([float(x) for x in source_sample_rate.split(",")], dtype=np.float64)
        if len(rates) != len(self.source_datasets):
            raise ValueError(
                "dia_align_aug_source_rates length must match source dataset count: "
                f"{len(rates)} vs {len(self.source_datasets)}"
            )
        self.source_sample_rate = rates / rates.sum()
        print(
            "DIAAlignAugDataset has "
            f"{len(self.source_datasets)} source datasets; "
            f"positive_prob={self.positive_prob}, "
            f"background_scale={self.background_scale}, "
            f"mask_dilation={self.mask_dilation}."
        )

    def _sample_source(self, idx):
        source_idx = np.random.choice(
            list(range(len(self.source_datasets))),
            p=self.source_sample_rate,
        )
        return self.source_datasets[source_idx][idx]

    def _pick_positive_target(self, sample):
        masks = sample[4]
        exists = sample[6]
        if masks.ndim != 3 or masks.shape[0] == 0:
            return None
        valid = []
        for idx in range(masks.shape[0]):
            exists_ok = idx < len(exists) and bool(exists[idx])
            if exists_ok and masks[idx].float().sum() > 0:
                valid.append(idx)
        if not valid:
            return None
        return random.choice(valid)

    def __getitem__(self, idx):
        sample = None
        target_idx = None
        for _ in range(20):
            sample = self._sample_source(idx)
            target_idx = self._pick_positive_target(sample)
            if target_idx is not None:
                break
        if sample is None or target_idx is None:
            raise RuntimeError("Unable to draw a positive mask for DIA alignment augmentation.")

        (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            sam_mask_shape,
            _exists,
            ref_ids,
            sent_ids,
        ) = sample

        target_mask = masks[target_idx].float()
        sam_input_shape = tuple(sam_mask_shape[0])
        image_mask = _mask_to_tensor_size(
            target_mask,
            image.shape,
            sam_input_shape=sam_input_shape,
            dilation=self.mask_dilation,
        )
        clip_mask = _mask_to_tensor_size(
            target_mask,
            image_clip.shape,
            sam_input_shape=image_clip.shape[-2:],
            dilation=max(0, self.mask_dilation // 2),
        )

        is_positive = random.random() < self.positive_prob
        if is_positive:
            aug_image = make_density_reduced_view(
                image,
                image_mask,
                background_scale=self.background_scale,
            )
            aug_clip = make_density_reduced_view(
                image_clip,
                clip_mask,
                background_scale=self.background_scale,
            )
            out_masks = target_mask[None, ...]
            exists = [True]
            answer = random.choice(self.answer_list).format(
                class_name=infer_phrase_from_conversation(conversations[target_idx])
            )
            variant = "focus_positive"
        else:
            aug_image = make_target_removed_view(image, image_mask)
            aug_clip = make_target_removed_view(image_clip, clip_mask)
            out_masks = torch.zeros(
                (0, int(target_mask.shape[0]), int(target_mask.shape[1])),
                dtype=target_mask.dtype,
            )
            exists = [False]
            answer = random.choice(self.neg_answer_list).format(
                class_name=infer_phrase_from_conversation(conversations[target_idx])
            )
            variant = "target_removed_negative"

        phrase = infer_phrase_from_conversation(conversations[target_idx])
        description = build_density_matched_description(phrase, target_mask)
        question = random.choice(self.question_list).format(description=description)
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)

        ref_id = ref_ids[target_idx] if isinstance(ref_ids, list) and ref_ids else ref_ids
        sent_id = sent_ids[target_idx] if isinstance(sent_ids, list) and sent_ids else sent_ids
        sample_id = f"dia_align_aug:{variant}:{ref_id}"
        return (
            sample_id,
            aug_image,
            aug_clip,
            [conv.get_prompt()],
            out_masks,
            [sam_mask_shape[0], (int(target_mask.shape[0]), int(target_mask.shape[1]))],
            exists,
            [sample_id],
            [sent_id],
        )
