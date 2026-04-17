from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch


def load_wikitext2_local(data_dir: str, split: str = "valid") -> str:
    path = Path(data_dir)
    if path.is_file():
        file_path = path
    else:
        file_path = path / f"{split}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"WikiText-2 file not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def load_c4_validation_local(jsonl_path: str, max_samples: int = 2000) -> str:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"C4 validation file not found: {jsonl_path}")
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text", "")).strip()
            if text:
                texts.append(text)
    return "\n\n".join(texts)


def build_ppl_blocks(
    tokenizer, text: str, block_size: int = 1024
) -> List[torch.Tensor]:
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].squeeze(0)
    if token_ids.numel() <= block_size:
        return [token_ids.unsqueeze(0)]

    blocks: List[torch.Tensor] = []
    start = 0
    while start + block_size <= token_ids.numel():
        blocks.append(token_ids[start : start + block_size].unsqueeze(0))
        start += block_size
    return blocks
