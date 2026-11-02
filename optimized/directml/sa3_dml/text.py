"""Prompt tokenization and T5Gemma encoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .session import SessionPool
from .specs import TEXT_TOKENS


@dataclass
class PromptEmbedding:
    hidden: np.ndarray
    mask: np.ndarray


class Tokenizer:
    """T5Gemma tokenizer: no special tokens, right-padded with id 0 to 256 tokens."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer as HFTokenizer
        self._tok = HFTokenizer.from_file(str(path))

    def __call__(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        ids = self._tok.encode(text, add_special_tokens=False).ids[:TEXT_TOKENS]
        input_ids = np.zeros((1, TEXT_TOKENS), np.int64)
        mask = np.zeros((1, TEXT_TOKENS), np.int64)
        input_ids[0, :len(ids)] = ids
        mask[0, :len(ids)] = 1
        return input_ids, mask


class TextEncoder:
    role = "text"

    def __init__(self, graph: Path, tokenizer: Path, pool: SessionPool):
        self.graph = graph
        self.tokenizer = Tokenizer(tokenizer)
        self.pool = pool

    def encode(self, text: str) -> PromptEmbedding:
        ids, mask = self.tokenizer(text)
        hidden = self.pool.get(self.role, self.graph).run({"input_ids": ids, "attention_mask": mask})[0]
        return PromptEmbedding(hidden.astype(np.float32), mask.astype(np.float32))
