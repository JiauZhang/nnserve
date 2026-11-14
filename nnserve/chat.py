import importlib
from pathlib import Path

from tokenizers import Tokenizer

from nnserve.engine import Engine, SamplingParams
from nnserve.handlers import ResponsesChunk, ResponsesOutput, UsageStats


def load_model_class(path):
    module_name, separator, class_name = path.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(f'model_class must be "module:ClassName", got {path!r}')
    return getattr(importlib.import_module(module_name), class_name)


def input_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    part["text"]
                    for part in content
                    if isinstance(part, dict) and part.get("type") in ("input_text", "text") and part.get("text")
                )
        return "\n".join(parts)
    return ""


class ChatHandler:
    def __init__(self, entry):
        manifest = entry.manifest
        model_class = load_model_class(manifest.get("model_class", ""))
        self.model = model_class.from_pretrained(entry.path)
        self.tokenizer = Tokenizer.from_file(str(Path(entry.path) / manifest.get("tokenizer", "tokenizer.json")))
        self.chat_template = manifest.get("chat_template")
        if not self.chat_template:
            raise ValueError(f"model {entry.id!r} manifest has no chat_template")
        self.eos_token_ids = tuple(manifest.get("eos_token_ids", ()))
        self.engine = Engine.from_model(self.model)

    def _prompt(self, request):
        text = input_text(request.input)
        if request.instructions:
            text = f"{request.instructions}\n{text}"
        return self.chat_template.replace("{{prompt}}", text)

    def _params(self, request):
        temperature = 1.0 if request.temperature is None else request.temperature
        return SamplingParams(
            max_new_tokens=request.max_output_tokens or 256,
            do_sample=temperature > 0,
            temperature=temperature or 1.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            eos_token_ids=self.eos_token_ids,
        )

    def respond(self, request, entry):
        input_ids = self.tokenizer.encode(self._prompt(request)).ids
        params = self._params(request)
        session = self.engine.create_session(capacity=len(input_ids) + params.max_new_tokens)
        try:
            result = self.engine.generate(session, input_ids, params)
            text = self.tokenizer.decode(result.tokens, skip_special_tokens=True)
        finally:
            self.engine.close_session(session)
        usage = UsageStats(input_tokens=len(input_ids), output_tokens=len(result.tokens))
        return ResponsesOutput(text, usage)

    def stream(self, request, entry):
        input_ids = self.tokenizer.encode(self._prompt(request)).ids
        params = self._params(request)
        session = self.engine.create_session(capacity=len(input_ids) + params.max_new_tokens)
        generated = []
        decoded = ""
        try:
            for chunk in self.engine.iter_generate(session, input_ids, params):
                if chunk.token is not None:
                    generated.append(chunk.token)
                    text = self.tokenizer.decode(generated, skip_special_tokens=True)
                    if text != decoded:
                        delta = text[len(decoded) :]
                        decoded = text
                        yield ResponsesChunk(delta)
                if chunk.finish_reason is not None:
                    yield ResponsesChunk(usage=UsageStats(input_tokens=len(input_ids), output_tokens=len(generated)))
        finally:
            self.engine.close_session(session)
