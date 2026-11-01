from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CompatModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class TextFormat(CompatModel):
    type: str = "text"
    name: str | None = None
    schema_definition: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None

    @property
    def is_structured(self):
        return self.type in ("json_object", "json_schema")


class TextConfig(CompatModel):
    format: TextFormat | None = None
    verbosity: str | None = None


class ReasoningConfig(CompatModel):
    effort: str | None = None
    summary: str | None = None


class ResponsesRequest(CompatModel):
    model: str
    input: str | list[dict[str, Any]] | None = None
    instructions: str | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    text: TextConfig | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    reasoning: ReasoningConfig | None = None
    user: str | None = None
    metadata: dict[str, str] | None = None
    store: bool = False
    truncation: str | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    conversation: Any = None
    background: bool = False


class InputTokensDetails(CompatModel):
    cached_tokens: int = 0
    cache_write_tokens: int = 0


class OutputTokensDetails(CompatModel):
    reasoning_tokens: int = 0


class ResponsesUsage(CompatModel):
    input_tokens: int = 0
    input_tokens_details: InputTokensDetails = Field(default_factory=InputTokensDetails)
    output_tokens: int = 0
    output_tokens_details: OutputTokensDetails = Field(default_factory=OutputTokensDetails)
    total_tokens: int = 0


class OutputText(CompatModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[Any] = Field(default_factory=list)
    logprobs: list[Any] = Field(default_factory=list)


class MessageItem(CompatModel):
    type: Literal["message"] = "message"
    id: str
    status: Literal["in_progress", "completed"] = "completed"
    role: Literal["assistant"] = "assistant"
    content: list[OutputText] = Field(default_factory=list)


class ResponsesError(CompatModel):
    code: str | None = None
    message: str
    param: str | None = None
    type: str = "server_error"


class IncompleteDetails(CompatModel):
    reason: str | None = None


class ResponsesResponse(CompatModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    completed_at: int | None = None
    status: str = "completed"
    error: ResponsesError | None = None
    incomplete_details: IncompleteDetails | None = None
    model: str
    previous_response_id: str | None = None
    instructions: str | None = None
    output: list[MessageItem] = Field(default_factory=list)
    output_text: str = ""
    tools: list[Any] = Field(default_factory=list)
    tool_choice: Any = "auto"
    truncation: str = "disabled"
    parallel_tool_calls: bool = True
    text: TextConfig = Field(default_factory=TextConfig)
    temperature: float = 1.0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    top_logprobs: int = 0
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    usage: ResponsesUsage = Field(default_factory=ResponsesUsage)
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    store: bool = False
    user: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class ModelInfo(CompatModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "nnm"
    task: str | None = None
    capabilities: list[str] = Field(default_factory=list)


SPEECH_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
TRANSCRIPTION_FORMATS = ("json", "text", "srt", "verbose_json", "vtt")


class SpeechRequest(CompatModel):
    model: str
    input: str = Field(min_length=1, max_length=4096)
    voice: str | dict[str, Any]
    instructions: str | None = None
    language: str | None = None
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: str | None = None

    @model_validator(mode="after")
    def check_format(self):
        if self.response_format not in SPEECH_FORMATS:
            raise ValueError(f"response_format must be one of {SPEECH_FORMATS}")
        if self.stream_format not in (None, "audio", "sse"):
            raise ValueError("stream_format must be audio or sse")
        return self


class TranscriptionRequest(CompatModel):
    model: str
    language: str | None = None
    prompt: str | None = None
    response_format: str = "json"
    temperature: float | None = None
    timestamp_granularities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_format(self):
        if self.response_format not in TRANSCRIPTION_FORMATS:
            raise ValueError(f"response_format must be one of {TRANSCRIPTION_FORMATS}")
        if any(value not in ("word", "segment") for value in self.timestamp_granularities):
            raise ValueError("timestamp_granularities must be word or segment")
        return self


class ModelList(CompatModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo] = Field(default_factory=list)


class OcrDocument(CompatModel):
    type: Literal["document_url", "image_url"]
    document_url: str | None = None
    image_url: str | None = None

    @model_validator(mode="after")
    def check_url(self):
        url = self.document_url if self.type == "document_url" else self.image_url
        if not url:
            raise ValueError(f"{self.type} requires a URL")
        if not url.startswith(("data:", "http://", "https://")):
            raise ValueError("OCR document URL must be a data URI or an http(s) URL")
        return self


class OcrRequest(CompatModel):
    model: str
    document: OcrDocument
    pages: str | list[int] | None = None
    include_image_base64: bool | None = None
    image_limit: int | None = None
    image_min_size: int | None = None
    table_format: str | None = None
    extract_header: bool | None = None
    extract_footer: bool | None = None
    include_blocks: bool | None = None
    confidence_scores_granularity: str | None = None
    bbox_annotation_format: dict[str, Any] | None = None
    document_annotation_format: dict[str, Any] | None = None
    document_annotation_prompt: str | None = None

    @model_validator(mode="after")
    def check_options(self):
        if self.table_format not in (None, "markdown", "html"):
            raise ValueError("table_format must be markdown or html")
        if self.confidence_scores_granularity not in (None, "page", "block", "word"):
            raise ValueError("confidence_scores_granularity must be page, block or word")
        return self


class OcrDimensions(CompatModel):
    dpi: float
    height: int
    width: int


class OcrImage(CompatModel):
    id: str
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float
    image_base64: str | None = None


class OcrTable(CompatModel):
    id: str
    content: str | None = None
    format: str | None = None


class BlockConfidenceScores(CompatModel):
    average_content_confidence_score: float | None = None
    minimum_content_confidence_score: float | None = None
    block_type_confidence_score: float | None = None


class PageConfidenceScores(CompatModel):
    average_page_confidence_score: float | None = None
    minimum_page_confidence_score: float | None = None


class WordConfidence(CompatModel):
    word: str
    confidence: float


class OcrBlock(CompatModel):
    type: str
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float
    content: str = ""
    confidence_scores: BlockConfidenceScores | None = None
    table_id: str | None = None
    image_id: str | None = None


class OcrPage(CompatModel):
    index: int
    markdown: str
    images: list[OcrImage] = Field(default_factory=list)
    tables: list[OcrTable] = Field(default_factory=list)
    blocks: list[OcrBlock] = Field(default_factory=list)
    header: str | None = None
    footer: str | None = None
    dimensions: OcrDimensions
    confidence_scores: PageConfidenceScores | None = None
    word_confidence_scores: list[WordConfidence] | None = None


class OcrUsageInfo(CompatModel):
    pages_processed: int
    doc_size_bytes: int


class OcrResponse(CompatModel):
    pages: list[OcrPage] = Field(default_factory=list)
    model: str
    document_annotation: str | None = None
    usage_info: OcrUsageInfo
    meta: dict[str, Any] | None = None
