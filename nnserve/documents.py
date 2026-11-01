import base64
import io
from dataclasses import dataclass

import httpx
from PIL import Image

from nnserve.schemas import OcrDocument

IMAGE_DPI = 200.0


@dataclass(frozen=True)
class DocumentSource:
    data: bytes
    media_type: str

    @property
    def size_bytes(self):
        return len(self.data)


@dataclass(frozen=True)
class PageImage:
    index: int
    width: int
    height: int
    dpi: float
    data: bytes
    media_type: str


def load_document(document: OcrDocument, timeout: float = 30.0) -> DocumentSource:
    url = document.document_url if document.type == "document_url" else document.image_url
    if url.startswith("data:"):
        header, payload = url.split(",", 1)
        media_type = header[5:].split(";")[0] or "application/octet-stream"
        return DocumentSource(base64.b64decode(payload), media_type)

    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    media_type = response.headers.get("content-type", "").split(";")[0] or "application/octet-stream"
    return DocumentSource(response.content, media_type)


def parse_page_selection(value) -> set[int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        pages = {value}
    elif isinstance(value, list):
        pages = {int(page) for page in value}
    else:
        text = str(value).strip()
        if "-" in text:
            start, end = (int(part) for part in text.split("-", 1))
            pages = set(range(start, end + 1))
        else:
            pages = {int(text)}
    if any(page < 0 for page in pages):
        raise ValueError("page selection must be non-negative")
    return pages


def is_pdf(source: DocumentSource) -> bool:
    return source.media_type == "application/pdf" or b"%PDF-" in source.data[:1024]


def render_pages(source: DocumentSource, selection: set[int] | None = None, dpi: float = IMAGE_DPI) -> list[PageImage]:
    if is_pdf(source):
        return _render_pdf(source, selection, dpi)
    return _render_image(source, selection)


def _render_pdf(source, selection, dpi):
    import pypdfium2 as pdfium  # noqa: PLC0415

    document = pdfium.PdfDocument(source.data)
    indices = range(len(document)) if selection is None else sorted(selection)
    if any(index >= len(document) for index in indices):
        document.close()
        raise ValueError("page selection exceeds the document page count")

    pages = []
    for index in indices:
        bitmap = document[index].render(scale=dpi / 72.0, grayscale=True)
        image = Image.fromarray(bitmap.to_numpy()).convert("L")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        pages.append(PageImage(index, image.width, image.height, dpi, buffer.getvalue(), "image/png"))
    document.close()
    return pages


def _render_image(source, selection):
    if selection is not None and selection != {0}:
        raise ValueError("an image document has only page 0")
    image = Image.open(io.BytesIO(source.data))
    image.load()
    dpi = float(image.info.get("dpi", (IMAGE_DPI, IMAGE_DPI))[0])
    return [PageImage(0, image.width, image.height, dpi, source.data, source.media_type)]
