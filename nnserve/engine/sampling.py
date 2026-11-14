from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    eos_token_ids: tuple[int, ...] = ()
    seed: int | None = None

    def __post_init__(self):
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


def select_token(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None = None):
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    scores = logits.float()

    if params.do_sample:
        scores = scores / params.temperature
        if params.top_k > 0:
            values, indices = torch.topk(scores, min(params.top_k, scores.shape[-1]), dim=-1)
            scores = torch.full_like(scores, float("-inf")).scatter(-1, indices, values)
        if params.top_p < 1:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
            keep = cumulative - torch.softmax(sorted_scores, dim=-1) < params.top_p
            sorted_scores = sorted_scores.masked_fill(~keep, float("-inf"))
            scores = torch.full_like(scores, float("-inf")).scatter(-1, sorted_indices, sorted_scores)
        probabilities = torch.softmax(scores, dim=-1)
        token = int(torch.multinomial(probabilities, num_samples=1, generator=generator))
    else:
        token = int(scores.argmax(dim=-1))

    return token, float(log_probs[token])
