import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from nnserve.app import create_app
from nnserve.config import ModelConfig, ServeConfig
from nnserve.documents import PageImage
from nnserve.ocr import IndraOcrHandler, geometry, layout
from nnserve.ocr.indra import to_ocr_page
from nnserve.ocr.order import _cluster, decode_order, line_features, order_xycut
from nnserve.ocr.pipeline import Line, Page
from nnserve.registry import ModelEntry
from nnserve.schemas import OcrRequest
from tests.helpers import PNG_DATA_URI


def make_request(**overrides):
    payload = {
        "model": "fake-ocr",
        "document": {"type": "image_url", "image_url": PNG_DATA_URI},
        "include_blocks": True,
        "table_format": "html",
        "confidence_scores_granularity": "page",
        "extract_header": True,
    }
    payload.update(overrides)
    return OcrRequest(**payload)


def make_page():
    return Page(
        lines=[
            Line([10, 10, 200, 30], "Title", 0.9),
            Line([10, 40, 200, 60], "Body", 0.8),
            Line([10, 80, 90, 100], "A", 0.7),
            Line([100, 80, 190, 100], "B", 0.6),
        ],
        blocks=[
            {"label": "Section-Header", "bbox": [10, 10, 200, 30], "text": "Title"},
            {"label": "Text", "bbox": [10, 40, 200, 60], "text": "Body"},
            {"label": "Table", "bbox": [10, 75, 200, 105], "text": "A\nB"},
            {"label": "Page-Header", "bbox": [10, 0, 200, 8], "text": "Header"},
        ],
        markdown="## Title\n\nBody",
    )


def test_ocr_page_maps_blocks_tables_and_header():
    page = make_page()
    image = PageImage(0, 200, 120, 200.0, b"", "image/png")
    result = to_ocr_page(page, image, make_request())

    assert result.index == 0
    assert result.markdown == "## Title\n\nBody"
    assert [block.type for block in result.blocks] == ["title", "text", "table", "header"]
    assert result.header == "Header"
    assert result.dimensions.width == 200
    assert result.confidence_scores.average_page_confidence_score == pytest.approx(0.75)
    assert len(result.tables) == 1
    assert result.tables[0].format == "html"
    assert result.tables[0].id == "tbl-0.html"
    assert "<table>" in result.tables[0].content


def test_ocr_page_markdown_tables():
    page = make_page()
    image = PageImage(0, 200, 120, 200.0, b"", "image/png")
    result = to_ocr_page(page, image, make_request(table_format="markdown"))

    assert result.tables[0].format == "markdown"
    assert result.tables[0].id == "tbl-0.md"
    assert result.tables[0].content.startswith("| ")


def test_ocr_page_without_blocks_or_tables():
    page = make_page()
    image = PageImage(0, 200, 120, 200.0, b"", "image/png")
    result = to_ocr_page(page, image, make_request(include_blocks=False, table_format=None, confidence_scores_granularity=None))

    assert result.blocks == []
    assert result.tables == []
    assert result.confidence_scores is None
    assert result.header == "Header"


def test_join_lines_joins_hyphenated_and_cjk():
    assert layout.join_lines(["tumori-", "genesis"]) == "tumorigenesis"
    assert layout.join_lines(["中文", "排版"]) == "中文排版"
    assert layout.join_lines(["hello", "world"]) == "hello world"


def test_page_markdown_can_drop_header():
    boxes = np.array([[0, 0, 100, 10], [0, 20, 100, 30]], np.float32)
    texts = ["header", "body"]
    cls = np.zeros((2, 10), np.float32)
    cls[0, 2] = 1.0
    cls[1, 0] = 1.0
    same = np.array([[0.0, -1.0], [-1.0, 0.0]], np.float32)

    assert "header" in layout.page_markdown(boxes, texts, [0, 1], cls, same, keep_header=True)
    assert "header" not in layout.page_markdown(boxes, texts, [0, 1], cls, same, keep_header=False)


def test_table_html_builds_header_and_cells():
    boxes = np.array([[0, 0, 40, 20], [60, 0, 100, 20], [0, 30, 40, 50], [60, 30, 100, 50]], np.float32)
    result = layout.table_html(boxes, ["A", "B", "1", "2"])

    assert "<th>A</th><th>B</th>" in result
    assert "<td>1</td><td>2</td>" in result


def test_html_table_to_markdown():
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"

    assert layout.html_table_to_markdown(html) == "| A | B |\n| --- | --- |\n| 1 | 2 |"


def test_line_features_shape():
    features = line_features(np.array([[0, 0, 100, 20]], np.float32), (200, 100), ["hello"])

    assert features.shape == (1, 16)


def test_decode_order_recovers_a_chain():
    scores = np.zeros((3, 3), np.float32)
    scores[0, 1] = scores[0, 2] = 3.0
    scores[1, 2] = 3.0

    assert decode_order(scores) == [0, 1, 2]


def test_cluster_splits_negative_logits():
    scores = np.array([[0, 2, -1, -1], [2, 0, -1, -1], [-1, -1, 0, 2], [-1, -1, 2, 0]], np.float32)

    assert _cluster(scores) == [[0, 1], [2, 3]]


def test_xycut_splits_columns():
    boxes = np.array([[0, 0, 40, 20], [0, 30, 40, 50], [60, 0, 100, 20], [60, 30, 100, 50]], np.float32)

    assert order_xycut(boxes) == [0, 1, 2, 3]


def test_crop_line_returns_ink_map():
    gray = np.zeros((100, 200), np.uint8)
    gray[20:40, 20:180] = 255
    crop = geometry.crop_line(gray, [20, 20, 180, 40], height=32)

    assert crop.shape[0] == 32
    assert crop.min() >= 0.0
    assert crop.max() <= 1.0


def test_decode_boxes_finds_a_region():
    out = np.full((3, 32, 64), -10.0, np.float32)
    out[0, 8:16, 10:30] = 10.0
    out[1] = 2.0
    out[2] = 2.0

    boxes = geometry.decode_boxes(out)

    assert boxes.shape == (1, 5)
    assert boxes[0, 2] > boxes[0, 0]


def synthetic_page():
    image = Image.new("L", (900, 400), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=48)
    draw.text((40, 60), "Indra OCR serving test", fill=0, font=font)
    draw.text((40, 160), "Second line of text", fill=0, font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return image, buffer.getvalue()


def test_indra_handler_reads_a_synthetic_page(model_path):
    if model_path is None:
        pytest.skip("--model-path required for the end-to-end OCR check")

    image, payload = synthetic_page()
    page = PageImage(0, image.width, image.height, 200.0, payload, "image/png")
    entry = ModelEntry(id="indra-ocr", path=str(model_path), task="ocr", capabilities=(), manifest={})
    handler = IndraOcrHandler(entry)
    request = OcrRequest(
        model="indra-ocr",
        document={"type": "image_url", "image_url": PNG_DATA_URI},
        include_blocks=True,
        confidence_scores_granularity="page",
    )
    result = handler.process(request, entry, [page])

    assert len(result.pages) == 1
    assert result.pages[0].dimensions.width == 900
    assert result.pages[0].markdown is not None


def test_indra_handler_through_the_http_api(model_path):
    if model_path is None:
        pytest.skip("--model-path required for the end-to-end OCR check")

    _, payload = synthetic_page()
    uri = "data:image/png;base64," + base64.b64encode(payload).decode()
    config = ServeConfig(models=[ModelConfig(id="indra-ocr", path=str(model_path), handler="nnserve.ocr:IndraOcrHandler")])
    client = TestClient(create_app(config))
    response = client.post(
        "/v1/ocr",
        json={
            "model": "indra-ocr",
            "document": {"type": "image_url", "image_url": uri},
            "include_blocks": True,
            "confidence_scores_granularity": "page",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "indra-ocr"
    assert body["pages"][0]["dimensions"]["width"] == 900
    assert body["pages"][0]["markdown"]
    assert body["pages"][0]["blocks"]
