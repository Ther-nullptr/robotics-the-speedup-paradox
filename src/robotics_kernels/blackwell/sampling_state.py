"""Device state shared by FP4 and FP8 batch-one sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch


@dataclass
class GreedyState:
    seen_tokens: torch.Tensor
    eos_tokens: torch.Tensor
    generated_count: torch.Tensor
    repetition_penalty: float
    min_new_tokens: int

    input_error: ClassVar[str] = "fused greedy sampling requires CUDA batch size one"

    @classmethod
    def create(
        cls,
        input_ids: torch.Tensor,
        vocab_size: int,
        eos_token_ids: torch.Tensor | list[int] | tuple[int, ...] | int | None,
        repetition_penalty: float,
        min_new_tokens: int,
    ) -> "GreedyState":
        if not input_ids.is_cuda or input_ids.shape[0] != 1:
            raise ValueError(cls.input_error)
        device = input_ids.device
        seen_tokens = torch.zeros(vocab_size, dtype=torch.uint8, device=device)
        seen_tokens[input_ids.reshape(-1)] = 1
        eos_tokens = torch.zeros_like(seen_tokens)
        if eos_token_ids is not None:
            eos = torch.as_tensor(
                eos_token_ids, dtype=torch.long, device=device
            ).reshape(-1)
            if eos.numel():
                eos_tokens[eos] = 1
        return cls(
            seen_tokens=seen_tokens,
            eos_tokens=eos_tokens,
            generated_count=torch.zeros(1, dtype=torch.int32, device=device),
            repetition_penalty=float(repetition_penalty),
            min_new_tokens=int(min_new_tokens),
        )

    def mark_sampled(self, token_ids: torch.Tensor) -> None:
        self.seen_tokens.scatter_(0, token_ids.reshape(-1), 1)
        self.generated_count.add_(1)
