import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    id: str
    path: str
    task: str | None = None
    capabilities: list[str] = field(default_factory=list)
    handler: str | None = None


@dataclass
class ServeConfig:
    models: list[ModelConfig] = field(default_factory=list)
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    pdf_dpi: float = 200.0
    request_timeout: float = 30.0

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text())
        models = [ModelConfig(**item) for item in data.get("models", [])]
        fields = {key: value for key, value in data.items() if key != "models"}
        return cls(models=models, **fields)
