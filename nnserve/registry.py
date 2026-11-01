import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_NAME = "nnm_weights.json"


@dataclass(frozen=True)
class ModelEntry:
    id: str
    path: str
    task: str
    capabilities: tuple[str, ...]
    manifest: dict[str, Any]


def read_manifest(path) -> dict[str, Any]:
    manifest_path = Path(path) / MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


class ModelRegistry:
    def __init__(self, entries):
        self._entries = {entry.id: entry for entry in entries}

    @classmethod
    def from_models(cls, models):
        entries = []
        for model in models:
            manifest = read_manifest(model.path)
            task = manifest.get("task") or model.task
            if not task:
                raise ValueError(f"model {model.id!r} has no task in its manifest or config")
            capabilities = tuple(sorted({*manifest.get("capabilities", []), *model.capabilities}))
            entries.append(ModelEntry(model.id, model.path, task, capabilities, manifest))
        return cls(entries)

    def get(self, model_id) -> ModelEntry:
        try:
            return self._entries[model_id]
        except KeyError:
            raise KeyError(f"model {model_id!r} is not registered") from None

    def list(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def __contains__(self, model_id):
        return model_id in self._entries
