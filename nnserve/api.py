# ruff: noqa: B008
import time
import uuid

import httpx
from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from nnserve.asr.audio import decode_audio
from nnserve.asr.format import render_transcription
from nnserve.documents import load_document, parse_page_selection, render_pages
from nnserve.handlers import ResponsesOutput, UsageStats
from nnserve.schemas import (
    TRANSCRIPTION_FORMATS,
    InputTokensDetails,
    MessageItem,
    ModelInfo,
    ModelList,
    OcrRequest,
    OcrResponse,
    OcrUsageInfo,
    OutputText,
    OutputTokensDetails,
    ResponsesError,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
    SpeechRequest,
    TextConfig,
    TextFormat,
    TranscriptionRequest,
)
from nnserve.streaming import EventSequence, encode_event
from nnserve.tts.audio import encode_audio

router = APIRouter()


class ApiError(Exception):
    def __init__(self, message, status_code=400, *, type="invalid_request_error", param=None, code=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.type = type
        self.param = param
        self.code = code

    def body(self):
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


def authorize(request: Request):
    api_key = request.app.state.config.api_key
    if api_key and request.headers.get("authorization") != f"Bearer {api_key}":
        raise ApiError("Invalid API key", 401, code="invalid_api_key")


def model_entry(request: Request, model_id):
    try:
        return request.app.state.registry.get(model_id)
    except KeyError:
        raise ApiError(f"model {model_id!r} is not registered", 404, param="model", code="model_not_found") from None


def model_handler(request: Request, entry):
    handler = request.app.state.handlers.get(entry.id)
    if handler is None:
        raise ApiError(f"model {entry.id!r} is not loaded", 503, type="server_error", code="model_not_loaded")
    return handler


def has_image(value):
    if isinstance(value, dict):
        return value.get("type") == "input_image" or any(has_image(item) for item in value.values())
    if isinstance(value, list):
        return any(has_image(item) for item in value)
    return False


def validate_responses(request: ResponsesRequest, entry):
    if request.tools:
        raise ApiError("tools are not supported", param="tools", code="unsupported_parameter")
    if request.tool_choice not in (None, "none"):
        raise ApiError("tool_choice is not supported", param="tool_choice", code="unsupported_parameter")
    format_type = request.text.format.type if request.text and request.text.format else "text"
    if format_type in ("json_object", "json_schema"):
        raise ApiError("structured text output is not supported", param="text.format", code="unsupported_parameter")
    if not request.input and not request.instructions:
        raise ApiError("input or instructions is required", param="input", code="missing_required_parameter")
    if has_image(request.input) and "vision" not in entry.capabilities:
        raise ApiError(f"model {entry.id!r} has no vision capability", param="input", code="unsupported_capability")


def text_config(request: ResponsesRequest):
    if request.text is None:
        return TextConfig(format=TextFormat())
    if request.text.format is None:
        return TextConfig(format=TextFormat(), verbosity=request.text.verbosity)
    return request.text


def build_usage(stats: UsageStats):
    return ResponsesUsage(
        input_tokens=stats.input_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=stats.cached_tokens),
        output_tokens=stats.output_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=stats.reasoning_tokens),
        total_tokens=stats.input_tokens + stats.output_tokens,
    )


def build_response(
    request: ResponsesRequest,
    output: ResponsesOutput,
    *,
    response_id=None,
    item_id=None,
    created_at=None,
    status="completed",
    error=None,
):
    now = created_at or int(time.time())
    item = MessageItem(
        id=item_id or f"msg_{uuid.uuid4().hex}",
        status="in_progress" if status == "in_progress" else "completed",
        content=[OutputText(text=output.text)],
    )
    return ResponsesResponse(
        id=response_id or f"resp_{uuid.uuid4().hex}",
        created_at=now,
        completed_at=None if status == "in_progress" else now,
        status=status,
        error=error,
        model=request.model,
        instructions=request.instructions,
        output=[item],
        output_text=output.text,
        text=text_config(request),
        temperature=request.temperature if request.temperature is not None else 1.0,
        top_p=request.top_p if request.top_p is not None else 1.0,
        truncation=request.truncation or "disabled",
        parallel_tool_calls=request.parallel_tool_calls if request.parallel_tool_calls is not None else True,
        usage=build_usage(output.usage),
        max_output_tokens=request.max_output_tokens,
        user=request.user,
        metadata=request.metadata or {},
    )


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/metrics")
def metrics(request: Request):
    entries = request.app.state.registry.list()
    lines = [
        "# HELP nnserve_models Registered models",
        "# TYPE nnserve_models gauge",
        f"nnserve_models {len(entries)}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


@router.get("/v1/models")
def list_models(request: Request):
    authorize(request)
    entries = request.app.state.registry.list()
    return ModelList(data=[ModelInfo(id=entry.id, task=entry.task, capabilities=list(entry.capabilities)) for entry in entries])


@router.get("/v1/models/{model_id}")
def get_model(model_id: str, request: Request):
    authorize(request)
    entry = model_entry(request, model_id)
    return ModelInfo(id=entry.id, task=entry.task, capabilities=list(entry.capabilities))


@router.post("/v1/responses")
def create_response(payload: ResponsesRequest, request: Request):
    authorize(request)
    entry = model_entry(request, payload.model)
    if entry.task not in ("chat", "vlm"):
        raise ApiError(
            f"model {entry.id!r} has task {entry.task!r}; use a task-specific endpoint",
            param="model",
            code="model_not_supported",
        )
    validate_responses(payload, entry)
    handler = model_handler(request, entry)

    if not payload.stream:
        return build_response(payload, handler.respond(payload, entry))

    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    created_at = int(time.time())
    return StreamingResponse(
        _response_stream(payload, entry, handler, response_id, item_id, created_at),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _response_stream(payload, entry, handler, response_id, item_id, created_at):
    sequence = EventSequence()
    partial = build_response(
        payload, ResponsesOutput(""), response_id=response_id, item_id=item_id, created_at=created_at, status="in_progress"
    )
    yield encode_event(sequence.next("response.created", response=partial.model_dump(mode="json")))
    yield encode_event(sequence.next("response.in_progress", response=partial.model_dump(mode="json")))
    yield encode_event(
        sequence.next(
            "response.output_item.added",
            output_index=0,
            item={"type": "message", "id": item_id, "status": "in_progress", "role": "assistant", "content": []},
        )
    )
    yield encode_event(
        sequence.next(
            "response.content_part.added",
            output_index=0,
            content_index=0,
            item_id=item_id,
            part={"type": "output_text", "text": "", "annotations": [], "logprobs": []},
        )
    )

    text = []
    usage = UsageStats()
    try:
        for chunk in handler.stream(payload, entry):
            if chunk.text:
                text.append(chunk.text)
                yield encode_event(
                    sequence.next(
                        "response.output_text.delta",
                        output_index=0,
                        content_index=0,
                        item_id=item_id,
                        delta=chunk.text,
                    )
                )
            if chunk.usage is not None:
                usage = chunk.usage
    except Exception as exc:  # noqa: BLE001
        failed = build_response(
            payload,
            ResponsesOutput("".join(text), usage),
            response_id=response_id,
            item_id=item_id,
            created_at=created_at,
            status="failed",
            error=ResponsesError(code="server_error", message=str(exc), type="server_error"),
        )
        yield encode_event(sequence.next("response.failed", response=failed.model_dump(mode="json")))
        return

    output = "".join(text)
    yield encode_event(
        sequence.next("response.output_text.done", output_index=0, content_index=0, item_id=item_id, text=output)
    )
    yield encode_event(
        sequence.next(
            "response.content_part.done",
            output_index=0,
            content_index=0,
            item_id=item_id,
            part={"type": "output_text", "text": output, "annotations": [], "logprobs": []},
        )
    )
    completed = build_response(
        payload, ResponsesOutput(output, usage), response_id=response_id, item_id=item_id, created_at=created_at
    )
    yield encode_event(
        sequence.next(
            "response.output_item.done",
            output_index=0,
            item=completed.output[0].model_dump(mode="json"),
        )
    )
    yield encode_event(sequence.next("response.completed", response=completed.model_dump(mode="json")))


@router.post("/v1/ocr")
def create_ocr(payload: OcrRequest, request: Request):
    authorize(request)
    entry = model_entry(request, payload.model)
    if entry.task != "ocr":
        raise ApiError(f"model {entry.id!r} has task {entry.task!r}", param="model", code="model_not_supported")
    if payload.bbox_annotation_format or payload.document_annotation_format or payload.document_annotation_prompt:
        raise ApiError("annotation formats are not supported", code="unsupported_parameter")
    handler = model_handler(request, entry)

    config = request.app.state.config
    try:
        source = load_document(payload.document, timeout=config.request_timeout)
        selection = parse_page_selection(payload.pages)
        pages = render_pages(source, selection, dpi=config.pdf_dpi)
    except ValueError as exc:
        raise ApiError(str(exc), code="invalid_document") from exc
    except httpx.HTTPError as exc:
        raise ApiError(f"could not fetch the document: {exc}", code="invalid_document") from exc

    if not pages:
        raise ApiError("no pages to process", code="invalid_document")

    result = handler.process(payload, entry, pages)
    return OcrResponse(
        pages=result.pages,
        model=payload.model,
        document_annotation=result.document_annotation,
        usage_info=OcrUsageInfo(pages_processed=len(result.pages), doc_size_bytes=source.size_bytes),
    )


@router.post("/v1/audio/speech")
def create_speech(payload: SpeechRequest, request: Request):
    authorize(request)
    entry = model_entry(request, payload.model)
    if entry.task != "tts":
        raise ApiError(f"model {entry.id!r} has task {entry.task!r}", param="model", code="model_not_supported")
    if payload.stream_format == "sse":
        raise ApiError("stream_format sse is not supported", param="stream_format", code="unsupported_parameter")
    handler = model_handler(request, entry)

    try:
        audio = handler.synthesize(payload, entry)
    except KeyError as exc:
        raise ApiError(str(exc), param="voice", code="voice_not_found") from exc
    except ValueError as exc:
        raise ApiError(str(exc), code="invalid_request") from exc

    content, media_type = encode_audio(audio.samples, audio.sample_rate, payload.response_format)
    return Response(content=content, media_type=media_type)


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float | None = Form(None),
    timestamp_granularities: list[str] | None = Form(None, alias="timestamp_granularities[]"),
    granularity_alias: list[str] | None = Form(None, alias="timestamp_granularities"),
):
    authorize(request)
    entry = model_entry(request, model)
    if entry.task != "asr":
        raise ApiError(f"model {entry.id!r} has task {entry.task!r}", param="model", code="model_not_supported")
    if response_format not in TRANSCRIPTION_FORMATS:
        raise ApiError(
            f"response_format must be one of {TRANSCRIPTION_FORMATS}",
            param="response_format",
            code="invalid_request",
        )
    granularities = timestamp_granularities or granularity_alias or []
    if any(value not in ("word", "segment") for value in granularities):
        raise ApiError("timestamp_granularities must be word or segment", code="invalid_request")
    handler = model_handler(request, entry)

    try:
        audio = decode_audio(await file.read())
    except ValueError as exc:
        raise ApiError(str(exc), code="invalid_audio") from exc
    except RuntimeError as exc:
        raise ApiError(str(exc), status_code=500, type="server_error", code="audio_decode_unavailable") from exc

    payload = TranscriptionRequest(
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=granularities,
    )
    try:
        result = handler.transcribe(payload, entry, audio)
    except ValueError as exc:
        raise ApiError(str(exc), code="invalid_request") from exc

    body, media_type = render_transcription(result, payload, len(audio) / 16000)
    if isinstance(body, dict):
        return JSONResponse(content=body)
    return PlainTextResponse(content=body, media_type=media_type)
