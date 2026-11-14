from pathlib import Path

import torch
from torch import nn

import nnserve.engine
from nnserve.engine import Engine, SamplingParams, SessionClosedError


class CyclingModel(nn.Module):
    def __init__(self, vocab_size=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.layers = nn.ModuleList([nn.Identity() for _ in range(2)])
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, cache=None):
        if cache is not None:
            keys = input_ids.float().unsqueeze(-1)
            for layer in cache:
                layer.update(keys, keys)
        next_ids = (input_ids + 1) % self.vocab_size
        logits = torch.full((*input_ids.shape, self.vocab_size), -5.0, device=input_ids.device)
        logits.scatter_(-1, next_ids.unsqueeze(-1), 5.0)
        return logits


class UniformModel(CyclingModel):
    def forward(self, input_ids, cache=None):
        return torch.zeros((*input_ids.shape, self.vocab_size), device=input_ids.device)


def make_engine(model=None):
    return Engine.from_model(model or CyclingModel())


def test_generate_stops_at_eos():
    engine = make_engine()
    session = engine.create_session()
    params = SamplingParams(max_new_tokens=10, eos_token_ids=(3,))

    result = engine.generate(session, [1], params)

    assert result.tokens == [2]
    assert result.finish_reason == "stop"
    assert -1e-3 < result.logprobs[0] <= 0


def test_generate_respects_the_length_limit():
    engine = make_engine()
    session = engine.create_session()

    result = engine.generate(session, [0], SamplingParams(max_new_tokens=4))

    assert result.tokens == [1, 2, 3, 4]
    assert len(result.logprobs) == 4
    assert result.finish_reason == "length"


def test_sampling_is_reproducible_with_a_seed():
    engine = make_engine(UniformModel())
    session = engine.create_session()
    params = SamplingParams(max_new_tokens=8, do_sample=True, seed=7)

    first = engine.generate(session, [0], params)
    session.close()
    second_session = engine.create_session()
    second = engine.generate(second_session, [0], params)

    assert first.tokens == second.tokens


def test_score_returns_one_logprob_per_target_token():
    engine = make_engine()

    scores = engine.score([1, 2, 3])

    assert len(scores) == 2
    assert all(-1e-3 < score <= 0 for score in scores)
    assert engine.score([1]) == []


def test_updating_weights_invalidates_session_caches():
    engine = make_engine()
    session = engine.create_session()
    engine.generate(session, [0], SamplingParams(max_new_tokens=2))
    assert session.cache.get_seq_length() > 0

    version = engine.update_weights(policy_version=1)

    assert version == 1
    assert engine.policy_version == 1
    assert session.policy_version == 1
    assert session.cache.is_empty()


def test_closed_session_is_rejected():
    engine = make_engine()
    session = engine.create_session()
    engine.close_session(session)

    try:
        engine.generate(session, [0])
    except SessionClosedError:
        pass
    else:
        raise AssertionError("closed sessions must not generate")


def test_engine_does_not_import_the_server_stack():
    root = Path(nnserve.engine.__file__).parent
    for path in root.glob("*.py"):
        source = path.read_text()
        assert "fastapi" not in source
        assert "nnserve.api" not in source
        assert "nnserve.schemas" not in source


def test_sleep_and_wake_keep_the_engine_usable():
    engine = make_engine()
    session = engine.create_session()
    engine.generate(session, [0], SamplingParams(max_new_tokens=2))

    engine.sleep()
    assert session.cache.is_empty()
    engine.wake()
    result = engine.generate(session, [0], SamplingParams(max_new_tokens=2))

    assert result.tokens == [1, 2]
