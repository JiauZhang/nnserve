import importlib
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from nnserve.api import ApiError, router
from nnserve.config import ServeConfig
from nnserve.registry import ModelRegistry
from nnserve.version import __version__

logger = logging.getLogger("nnserve")


async def api_error_handler(request: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status_code, content=exc.body())


async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first = errors[0] if errors else {}
    location = [str(part) for part in first.get("loc", ())]
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(first.get("msg", "Invalid request")),
                "type": "invalid_request_error",
                "param": location[-1] if location else None,
                "code": "invalid_request",
            }
        },
    )


async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("unhandled service error")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "server_error", "param": None, "code": "server_error"}},
    )


def load_handler(spec, entry):
    module_name, separator, class_name = spec.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(f'handler must be "module:ClassName", got {spec!r}')
    return getattr(importlib.import_module(module_name), class_name)(entry)


def build_handlers(config, registry):
    handlers = {}
    for model in config.models:
        if model.handler:
            handlers[model.id] = load_handler(model.handler, registry.get(model.id))
    return handlers


def create_app(config: ServeConfig | None = None, handlers=None) -> FastAPI:
    config = config or ServeConfig()
    app = FastAPI(title="nnserve", version=__version__)
    app.state.config = config
    app.state.registry = ModelRegistry.from_models(config.models)
    app.state.handlers = dict(handlers) if handlers is not None else build_handlers(config, app.state.registry)
    app.include_router(router)
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
    return app
