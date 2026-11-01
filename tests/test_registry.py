import json

import pytest

from nnserve.app import create_app
from nnserve.config import ModelConfig, ServeConfig
from nnserve.registry import ModelRegistry


def make_manifest(path, payload):
    path.mkdir()
    (path / "nnm_weights.json").write_text(json.dumps(payload))
    return path


def test_manifest_task_and_capabilities(tmp_path):
    path = make_manifest(
        tmp_path / "model",
        {"version": 1, "sources": [], "weights": ["model.safetensors"], "task": "ocr", "capabilities": ["vision"]},
    )
    registry = ModelRegistry.from_models([ModelConfig(id="model", path=str(path), capabilities=["audio"])])

    entry = registry.get("model")
    assert entry.task == "ocr"
    assert entry.capabilities == ("audio", "vision")
    assert "model" in registry


def test_config_task_is_a_fallback(tmp_path):
    path = make_manifest(tmp_path / "model", {"version": 1, "sources": [], "weights": ["model.safetensors"]})
    registry = ModelRegistry.from_models([ModelConfig(id="model", path=str(path), task="chat")])

    assert registry.get("model").task == "chat"


def test_missing_task_is_rejected(tmp_path):
    path = make_manifest(tmp_path / "model", {"version": 1, "sources": [], "weights": ["model.safetensors"]})

    with pytest.raises(ValueError):
        ModelRegistry.from_models([ModelConfig(id="model", path=str(path))])


def test_unknown_model_raises(tmp_path):
    path = make_manifest(tmp_path / "model", {"version": 1, "sources": [], "weights": [], "task": "chat"})
    registry = ModelRegistry.from_models([ModelConfig(id="model", path=str(path))])

    with pytest.raises(KeyError):
        registry.get("missing")


def test_handler_is_built_from_config(tmp_path):
    path = make_manifest(tmp_path / "model", {"version": 1, "sources": [], "weights": [], "task": "chat"})
    config = ServeConfig(models=[ModelConfig(id="model", path=str(path), handler="tests.handlers:EntryHandler")])
    app = create_app(config)

    assert app.state.handlers["model"].entry.id == "model"
