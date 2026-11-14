from collections.abc import Iterator
from dataclasses import dataclass
import threading

import torch
from nnm.cache import Cache, KVCache

from .sampling import SamplingParams, select_token
from .session import Session


class EngineError(RuntimeError):
    pass


class SessionClosedError(EngineError):
    pass


@dataclass(frozen=True)
class Generation:
    tokens: list[int]
    logprobs: list[float]
    finish_reason: str


@dataclass(frozen=True)
class GenerationChunk:
    token: int | None = None
    logprob: float = 0.0
    finish_reason: str | None = None


class Engine:
    def __init__(self, model, *, cache_factory=None, device=None):
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()
        self.cache_factory = cache_factory or self._default_cache_factory
        self.policy_version = 0
        self._sessions: set[Session] = set()
        self._lock = threading.RLock()
        self._sleep_device = None

    @classmethod
    def from_model(cls, model, *, cache_factory=None, device=None):
        return cls(model, cache_factory=cache_factory, device=device)

    def _default_cache_factory(self):
        root = getattr(self.model, "backbone", self.model)
        layers = getattr(root, "layers", None)
        if layers is None:
            raise EngineError("cache_factory is required for models without a layers attribute")
        layer_types = getattr(root, "layer_types", None)
        if layer_types is None:
            return Cache([KVCache() for _ in layers])
        config = getattr(root, "config", None)
        window_size = getattr(config, "sliding_window", None)
        state_size = getattr(config, "conv_L_cache", 3)
        return Cache.from_layer_types(layer_types, window_size=window_size, state_size=state_size)

    def create_session(self, policy_version=None) -> Session:
        session = Session(
            cache=self.cache_factory(),
            policy_version=self.policy_version if policy_version is None else policy_version,
        )
        self._sessions.add(session)
        return session

    def close_session(self, session: Session):
        session.close()
        self._sessions.discard(session)

    def generate(self, session: Session, input_ids=None, params: SamplingParams | None = None) -> Generation:
        tokens = []
        logprobs = []
        finish_reason = "length"
        for chunk in self.iter_generate(session, input_ids, params):
            if chunk.token is not None:
                tokens.append(chunk.token)
                logprobs.append(chunk.logprob)
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
        return Generation(tokens, logprobs, finish_reason)

    def prefill(self, session: Session, token_ids):
        if session.closed:
            raise SessionClosedError("session is closed")
        with torch.no_grad():
            logits = self._logits(self._as_ids(token_ids), session.cache)
        session.last_logits = logits[:, -1]
        return session.last_logits

    def iter_generate(
        self, session: Session, input_ids=None, params: SamplingParams | None = None
    ) -> Iterator[GenerationChunk]:
        params = params or SamplingParams()
        if session.closed:
            raise SessionClosedError("session is closed")
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(params.seed)

        if input_ids is not None:
            logits = self.prefill(session, input_ids).squeeze(0)
        elif session.last_logits is not None:
            logits = session.last_logits.squeeze(0)
        else:
            raise EngineError("input_ids are required when the session has no cached logits")

        with torch.no_grad():
            for step in range(params.max_new_tokens):
                token, logprob = select_token(logits, params, generator)
                if token in params.eos_token_ids:
                    yield GenerationChunk(finish_reason="stop")
                    return
                yield GenerationChunk(token=token, logprob=logprob)
                if step + 1 == params.max_new_tokens:
                    yield GenerationChunk(finish_reason="length")
                    return
                logits = self._logits(self._as_ids([token]), session.cache)[:, -1].squeeze(0)
                session.last_logits = logits

    @torch.no_grad()
    def score(self, token_ids, cache: Cache | None = None) -> list[float]:
        ids = self._as_ids(token_ids)
        if ids.shape[1] < 2:
            return []
        logits = self._logits(ids, cache)
        log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = ids[:, 1:].unsqueeze(-1)
        return log_probs.gather(-1, targets).squeeze(-1)[0].tolist()

    def update_weights(self, state_dict=None, policy_version=None):
        with self._lock:
            if state_dict is not None:
                self.model.load_state_dict(state_dict, strict=True)
            if policy_version is None:
                policy_version = self.policy_version + 1
            if policy_version != self.policy_version:
                self.policy_version = policy_version
                for session in self._sessions:
                    session.reset_cache()
                    session.policy_version = policy_version
            return self.policy_version

    def sleep(self):
        with self._lock:
            if self._sleep_device is None:
                self._sleep_device = self.device
            self.model.to("cpu")
            for session in self._sessions:
                session.reset_cache()

    def wake(self):
        with self._lock:
            if self._sleep_device is not None:
                self.model.to(self._sleep_device)
                self.device = self._sleep_device
                self._sleep_device = None

    def close(self):
        for session in list(self._sessions):
            session.close()
        self._sessions.clear()

    def _as_ids(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.to(self.device)
        else:
            ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        return ids.reshape(1, -1)

    def _logits(self, input_ids, cache):
        return self.model(input_ids, cache=cache)
