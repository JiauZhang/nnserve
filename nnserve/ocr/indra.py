import threading

import cv2
import numpy as np

from nnserve.documents import PageImage
from nnserve.handlers import OcrResult
from nnserve.ocr import layout
from nnserve.ocr.pipeline import IndraOcrPipeline, Page
from nnserve.schemas import (
    BlockConfidenceScores,
    OcrBlock,
    OcrDimensions,
    OcrPage,
    OcrTable,
    PageConfidenceScores,
    WordConfidence,
)

LABEL_TYPES = {
    "Text": "text",
    "Section-Header": "title",
    "Page-Header": "header",
    "Page-Footer": "footer",
    "List-Group": "list",
    "Table": "table",
    "Caption": "caption",
    "Footnote": "aside_text",
    "Table-Of-Contents": "text",
    "Form": "text",
    "Picture": "image",
}


def _inside(line_box, block_box):
    center_x = (line_box[0] + line_box[2]) / 2
    center_y = (line_box[1] + line_box[3]) / 2
    return block_box[0] <= center_x <= block_box[2] and block_box[1] <= center_y <= block_box[3]


def _block_lines(page, block_box):
    return [line for line in page.lines if _inside(line.box, block_box)]


def _confidence(lines):
    scores = [line.conf for line in lines if line.text.strip()]
    if not scores:
        return None
    return sum(scores) / len(scores), min(scores)


def _block_confidence(page, block_box):
    values = _confidence(_block_lines(page, block_box))
    if values is None:
        return None
    average, minimum = values
    return BlockConfidenceScores(average_content_confidence_score=average, minimum_content_confidence_score=minimum)


def _page_confidence(page):
    values = _confidence(page.lines)
    if values is None:
        return None
    average, minimum = values
    return PageConfidenceScores(average_page_confidence_score=average, minimum_page_confidence_score=minimum)


def _word_confidences(page):
    return [WordConfidence(word=word, confidence=line.conf) for line in page.lines for word in line.text.split()]


def _table(page, block_box, table_format, counter):
    lines = _block_lines(page, block_box)
    if len(lines) < 2:
        return None, None
    boxes = np.array([line.box for line in lines], np.float32).reshape(-1, 4)
    texts = [line.text for line in lines]
    html_table = layout.table_html(boxes, texts)
    if table_format == "markdown":
        table_id = f"tbl-{counter}.md"
        return table_id, OcrTable(id=table_id, content=layout.html_table_to_markdown(html_table), format="markdown")
    table_id = f"tbl-{counter}.html"
    return table_id, OcrTable(id=table_id, content=html_table, format="html")


def to_ocr_page(page: Page, image: PageImage, request) -> OcrPage:
    granularity = request.confidence_scores_granularity
    include_blocks = bool(request.include_blocks)
    blocks = []
    tables = []
    header_texts = []
    footer_texts = []

    for block in page.blocks:
        label = block["label"]
        block_box = [float(value) for value in block["bbox"]]
        block_type = LABEL_TYPES.get(label, "text")
        content = block["text"]
        table_id = None

        if label == "Page-Header":
            header_texts.append(content)
        if label == "Page-Footer":
            footer_texts.append(content)

        if label == "Table" and request.table_format in ("html", "markdown"):
            table_id, table = _table(page, block_box, request.table_format, len(tables))
            if table is not None:
                content = table.content
                tables.append(table)

        if include_blocks:
            blocks.append(
                OcrBlock(
                    type=block_type,
                    top_left_x=block_box[0],
                    top_left_y=block_box[1],
                    bottom_right_x=block_box[2],
                    bottom_right_y=block_box[3],
                    content=content,
                    confidence_scores=_block_confidence(page, block_box) if granularity in ("block", "word") else None,
                    table_id=table_id,
                )
            )

    return OcrPage(
        index=image.index,
        markdown=page.markdown or page.text,
        images=[],
        tables=tables,
        blocks=blocks,
        header="\n".join(header_texts) if request.extract_header and header_texts else None,
        footer="\n".join(footer_texts) if request.extract_footer and footer_texts else None,
        dimensions=OcrDimensions(dpi=image.dpi, height=image.height, width=image.width),
        confidence_scores=_page_confidence(page) if granularity else None,
        word_confidence_scores=_word_confidences(page) if granularity == "word" else None,
    )


class IndraOcrHandler:
    def __init__(self, entry):
        self.pipeline = IndraOcrPipeline(entry.path)
        self.lock = threading.Lock()

    def process(self, request, entry, pages):
        results = []
        for image in pages:
            gray = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"page {image.index} is not a readable image")
            with self.lock:
                page = self.pipeline.read(
                    gray,
                    keep_header=not bool(request.extract_header),
                    keep_footer=not bool(request.extract_footer),
                )
            results.append(to_ocr_page(page, image, request))
        return OcrResult(pages=results)
