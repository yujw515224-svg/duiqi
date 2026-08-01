import os
import random

import cv2
import numpy as np
import torch

from model.llava import conversation as conversation_lib
from .base_dataset import BaseDataset
from .qa_template import SHORT_ANSWER_TEMPLATE, SHORT_QUESTION_TEMPLATE


class RefSegRSDataset(BaseDataset):
    """RefSegRS loader for LISAt referring-segmentation training/eval.

    Expected layout:
      <base_image_dir>/RefSegRS/images/<image_id>.tif
      <base_image_dir>/RefSegRS/masks/<image_id>.tif
      <base_image_dir>/RefSegRS/output_phrase_<split>.txt
    Each phrase line is: "<image_id> <referring expression>".
    """

    def __init__(
        self,
        base_image_dir,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        image_size: int = 224,
        num_classes_per_sample: int = 1,
        split: str = "train",
        is_train: bool | None = None,
    ):
        super().__init__(vision_tower, samples_per_epoch, image_size)
        self.base_image_dir = base_image_dir
        self.num_classes_per_sample = num_classes_per_sample
        self.split = split
        self.is_train = (split == "train") if is_train is None else is_train
        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.answer_list = SHORT_ANSWER_TEMPLATE

        root = os.path.join(base_image_dir, "RefSegRS")
        if not os.path.isdir(root):
            alt_root = os.path.join(base_image_dir, "rs_ref_seg", "RefSegRS")
            if os.path.isdir(alt_root):
                root = alt_root
        self.root = root
        self.image_root = os.path.join(root, "images")
        self.mask_root = os.path.join(root, "masks")
        phrase_path = os.path.join(root, f"output_phrase_{split}.txt")
        if not os.path.exists(phrase_path):
            raise FileNotFoundError(f"RefSegRS phrase file not found: {phrase_path}")

        samples = []
        with open(phrase_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                image_id, phrase = parts[0], parts[1].strip()
                image_path = os.path.join(self.image_root, f"{image_id}.tif")
                mask_path = os.path.join(self.mask_root, f"{image_id}.tif")
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    samples.append((image_id, phrase, image_path, mask_path, line_no))
        if not samples:
            raise RuntimeError(f"No usable RefSegRS samples found under {root}")
        self.samples = samples
        mode = "train" if self.is_train else "eval"
        print(f"RefSegRS({split}, {mode}) has {len(self.samples)} usable samples.")

    def __len__(self):
        if self.is_train:
            return self.samples_per_epoch
        if self.samples_per_epoch and self.samples_per_epoch > 0:
            return min(int(self.samples_per_epoch), len(self.samples))
        return len(self.samples)

    def _load_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return (mask > 0).astype(np.float32)

    def __getitem__(self, idx):
        if self.is_train:
            image_id, phrase, image_path, mask_path, line_no = random.choice(self.samples)
            question_template = random.choice(self.short_question_list)
            answer_template = random.choice(self.answer_list)
        else:
            image_id, phrase, image_path, mask_path, line_no = self.samples[idx % len(self.samples)]
            question_template = self.short_question_list[0]
            answer_template = self.answer_list[0]

        image, image_clip, sam_input_shape = self.load_and_preprocess_image(image_path)
        mask = self._load_mask(mask_path)
        masks = torch.from_numpy(mask[None, ...])

        phrase = phrase.lower()
        question = question_template.format(class_name=phrase)
        answer = answer_template.format(class_name=phrase)
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)

        sam_mask_shape = [sam_input_shape, (int(mask.shape[0]), int(mask.shape[1]))]
        return (
            image_path,
            image,
            image_clip,
            [conv.get_prompt()],
            masks,
            sam_mask_shape,
            [True],
            [image_id],
            [line_no],
        )
