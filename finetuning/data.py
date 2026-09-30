"""
data.py — plain Qwen3.5-VL VQA dataset (no VGGT / 4D-Humans fusion).

Support code for finetuning/train_qwen35.py (paper Section 5 component-wise
fine-tuning). This consolidates the plain-path helpers and dataset used by the
staged fine-tuning experiment; the geometric/human feature-fusion pipeline is
intentionally excluded from the public repository.

Each sample returns:
  - input_ids, attention_mask, pixel_values, image_grid_thw
  - labels (only the assistant response is trained; the rest is masked to -100)
  - mm_token_type_ids (only when the processor returns it)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


# ---------------------------------------------------------------------------
# Prompt construction (verbatim from the research code's plain path)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert in urban outdoor scene analysis. "
    "Please answer the question based on the image."
)
MC_SUFFIX = (
    "\n\nChoose the letter corresponding to your answer. "
    "Your output must be a single letter only.\nAnswer:"
)
COUNT_SUFFIX = "\nYour output must be a single integer only. If the count is 11 or more, output '11 or more'.\nAnswer:"


def _build_mc_user(question: str, options: list[str]) -> str:
    letters = ["A", "B", "C", "D", "E"]
    opts_str = "\n".join(f"{letters[i]}: {o}" for i, o in enumerate(options))
    return f"Question: {question}\n\nOptions:\n{opts_str}{MC_SUFFIX}"


def _build_count_user(question: str) -> str:
    return f"Question: {question}{COUNT_SUFFIX}"


def build_image_lookup(image_dir: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for p in image_dir.rglob("*.jpg"):
        lookup[p.stem] = p
    for p in image_dir.rglob("*.png"):
        lookup[p.stem] = p
    return lookup


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PlainVQADataset(Dataset):
    """Plain Qwen3.5-VL VQA dataset (no fusion features).

    Parameters
    ----------
    jsonl_path     : path to the training annotations jsonl
    image_dir      : root directory of frame images
    processor      : Qwen3.5-VL AutoProcessor
    max_len        : max token length (kept for compatibility; not used to truncate)
    is_train       : if True, shuffle multiple-choice options at train time
    sample_indices : index list for the train/val split
    """

    def __init__(
        self,
        jsonl_path: str,
        image_dir: str,
        processor: AutoProcessor,
        max_len: int = 4096,
        is_train: bool = True,
        sample_indices: list[int] | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.max_len = max_len
        self.is_train = is_train

        with open(jsonl_path) as f:
            all_samples = [json.loads(l) for l in f if l.strip()]

        if sample_indices is not None:
            self.samples = [all_samples[i] for i in sample_indices]
        else:
            self.samples = all_samples

        print(f"PlainVQADataset ({'train' if is_train else 'val'}): "
              f"{len(self.samples)} samples")
        print("Building image lookup…")
        self.lookup = build_image_lookup(self.image_dir)
        print(f"  {len(self.lookup)} images indexed")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        stem = Path(sample["image"]).stem
        img_path = self.lookup.get(stem)
        if img_path is None:
            img_path = next(iter(self.lookup.values()))

        img = Image.open(img_path).convert("RGB")

        answer = str(sample.get("answer", ""))
        options = list(sample.get("options", []))
        category = sample.get("category", "")
        question = sample.get("question", "")

        # Option shuffle (train only) + convert the answer to its letter
        if options and category != "counting" and answer in options:
            if self.is_train:
                shuffled = options[:]
                random.shuffle(shuffled)
                answer = chr(65 + shuffled.index(answer))
                options = shuffled
            else:
                answer = chr(65 + options.index(answer))

        if category == "counting" or not options:
            user_text = _build_count_user(question)
        else:
            user_text = _build_mc_user(question, options)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": user_text}],
            },
            {"role": "assistant", "content": answer},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.squeeze(0)

        # labels: mask everything up to the start of the assistant response with -100
        labels = input_ids.clone()
        assistant_token = self.processor.tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )
        seq = input_ids.tolist()
        mask_until = 0
        for i in range(len(seq) - len(assistant_token) + 1):
            if seq[i: i + len(assistant_token)] == assistant_token:
                mask_until = i + len(assistant_token) + 1
                break
        labels[:mask_until] = -100

        result: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
        }
        if mm_token_type_ids is not None:
            result["mm_token_type_ids"] = mm_token_type_ids
        return result


def plain_collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Pad variable-length input_ids/labels and concatenate pixel_values."""
    pad_id = 0
    max_len = max(x["input_ids"].size(0) for x in batch)

    input_ids_list, attn_list, labels_list, mm_type_list = [], [], [], []
    has_mm = "mm_token_type_ids" in batch[0]

    for x in batch:
        pad = max_len - x["input_ids"].size(0)
        input_ids_list.append(torch.cat([
            x["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)
        ]))
        attn_list.append(torch.cat([
            x["attention_mask"], torch.zeros(pad, dtype=torch.long)
        ]))
        labels_list.append(torch.cat([
            x["labels"], torch.full((pad,), -100, dtype=torch.long)
        ]))
        if has_mm:
            mm_type_list.append(torch.cat([
                x["mm_token_type_ids"], torch.zeros(pad, dtype=torch.long)
            ]))

    pixel_values = torch.cat([x["pixel_values"] for x in batch], dim=0)
    image_grid_thw = torch.cat([x["image_grid_thw"] for x in batch], dim=0)

    result = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_list),
        "labels": torch.stack(labels_list),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    if has_mm:
        result["mm_token_type_ids"] = torch.stack(mm_type_list)
    return result
