import html
import itertools
import re
from html.parser import HTMLParser

import numpy as np

from nnserve.ocr.order import BLOCK_CLASSES, _cluster

HEADER = BLOCK_CLASSES.index("Page-Header")
FOOTER = BLOCK_CLASSES.index("Page-Footer")
SECTION = BLOCK_CLASSES.index("Section-Header")
LIST = BLOCK_CLASSES.index("List-Group")
TABLE = BLOCK_CLASSES.index("Table")

_WHOLE_MATH = re.compile(r"^\$[^$]+\$$")
_ITEM = re.compile(r"^\s*([•\-–▪◦*·→]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)")


def _cjk(character):
    code = ord(character)
    return (
        0x3000 <= code <= 0x30FF
        or 0x3400 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF00 <= code <= 0xFFEF
        or 0x20000 <= code <= 0x2FA1F
    )


def join_lines(texts):
    out = ""
    for value in texts:
        value = value.strip()
        if not value:
            continue
        if len(out) > 1 and out.endswith("-") and out[-2].isalpha() and value[0].isalpha() and value[0].islower():
            out = out[:-1] + value
        elif out and _cjk(out[-1]) and _cjk(value[0]):
            out += value
        else:
            out = f"{out} {value}" if out else value
    return out


def split_blocks(order, same):
    blocks = []
    for index in order:
        if blocks and same[blocks[-1][-1], index] > 0:
            blocks[-1].append(index)
        else:
            blocks.append([index])
    return blocks


def _rows(boxes):
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2
    height = float(np.median(boxes[:, 3] - boxes[:, 1]))
    rows = []
    for index in np.argsort(center_y):
        if rows and abs(center_y[index] - np.mean(center_y[rows[-1]])) < 0.5 * height:
            rows[-1].append(int(index))
        else:
            rows.append([int(index)])
    return [sorted(row, key=lambda item: boxes[item, 0]) for row in rows]


def _merge_wrapped(rows, boxes):
    if len(rows) < 3:
        return rows
    height = float(np.median(boxes[:, 3] - boxes[:, 1]))
    gaps = [float(boxes[nxt, 1].min() - boxes[cur, 3].max()) for cur, nxt in itertools.pairwise(rows)]
    median = float(np.median(gaps))
    out = [list(rows[0])]
    previous = rows[0]
    for gap, row in zip(gaps, rows[1:]):
        if gap < 0.5 * median and gap < 0.6 * height and len(row) < len(previous):
            out[-1] += row
        else:
            out.append(list(row))
        previous = row
    return out


def table_html(boxes, texts):
    rows = _rows(boxes)
    span_x = boxes[:, 2].max() - boxes[:, 0].min()
    if (
        len(rows) > 2
        and len(rows[0]) == 1
        and max(len(row) for row in rows) > 1
        and boxes[rows[0][0], 2] - boxes[rows[0][0], 0] > 0.6 * span_x
    ):
        keep = [index for index in range(len(texts)) if index != rows[0][0]]
        return texts[rows[0][0]] + "\n\n" + table_html(boxes[keep], [texts[index] for index in keep])

    width = max(len(row) for row in rows)
    full = [row for row in rows if len(row) == width]
    rows = _merge_wrapped(rows, boxes)
    bands = np.array(
        [
            [np.median([boxes[row[column], 0] for row in full]), np.median([boxes[row[column], 2] for row in full])]
            for column in range(width)
        ]
    )
    edges = [-np.inf] + [(bands[column, 1] + bands[column + 1, 0]) / 2 for column in range(width - 1)] + [np.inf]

    out = ["<table>"]
    for row_index, row in enumerate(rows):
        tag = "th" if row_index == 0 and len(rows) > 1 else "td"
        cells = {}
        for index in sorted(row, key=lambda item: (boxes[item, 1], boxes[item, 0])):
            x0, x1 = boxes[index, 0], boxes[index, 2]
            first = int(np.searchsorted(edges, x0, side="right") - 1)
            last = int(np.searchsorted(edges, x1, side="right") - 1)
            span = [
                column
                for column in range(max(first, 0), min(last, width - 1) + 1)
                if min(x1, bands[column, 1]) - max(x0, bands[column, 0]) > 0.5 * (bands[column, 1] - bands[column, 0])
            ] or [min(max(first, 0), width - 1)]
            start = span[0]
            previous = cells.get(start)
            text = texts[index] if previous is None else previous[1] + " " + texts[index]
            cells[start] = (len(span) if previous is None else max(previous[0], len(span)), text)

        cells_html, column = [], 0
        while column < width:
            if column in cells:
                count, text = cells[column]
                attribute = f' colspan="{count}"' if count > 1 else ""
                cells_html.append(f"<{tag}{attribute}>{html.escape(text, quote=False)}</{tag}>")
                column += count
            else:
                cells_html.append(f"<{tag}></{tag}>")
                column += 1
        out.append("<tr>" + "".join(cells_html) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def heads_plausible(boxes, table_logits):
    cells = _cluster(table_logits[2])
    bad = 0
    for cell in cells:
        if len(cell) < 2:
            continue
        x0, x1 = boxes[cell, 0], boxes[cell, 2]
        if (np.minimum.outer(x1, x1) - np.maximum.outer(x0, x0)).min() < 0:
            bad += 1
    return bad <= 0.1 * len(cells)


def table_html_heads(boxes, texts, table_logits):
    cells = _cluster(table_logits[2])
    cell_boxes = np.array(
        [[boxes[cell, 0].min(), boxes[cell, 1].min(), boxes[cell, 2].max(), boxes[cell, 3].max()] for cell in cells]
    )
    cell_texts = [
        " ".join(texts[index] for index in sorted(cell, key=lambda item: (boxes[item, 1], boxes[item, 0]))) for cell in cells
    ]
    count = len(cells)

    def cell_logits(level):
        out = np.zeros((count, count), np.float32)
        for left in range(count):
            for right in range(left + 1, count):
                out[left, right] = out[right, left] = table_logits[level][np.ix_(cells[left], cells[right])].mean()
        return out

    rows = sorted(_cluster(cell_logits(0)), key=lambda group: cell_boxes[group, 1].min())
    columns = sorted(_cluster(cell_logits(1)), key=lambda group: cell_boxes[group, 0].min())
    row_of = {cell: row for row, group in enumerate(rows) for cell in group}
    column_of = {cell: column for column, group in enumerate(columns) for cell in group}
    centers = [float(np.median((cell_boxes[group, 0] + cell_boxes[group, 2]) / 2)) for group in columns]
    height = float(np.median(boxes[:, 3] - boxes[:, 1]))

    grid = {}
    for cell in range(count):
        column = column_of[cell]
        span = 1
        while (
            column + span < len(columns)
            and cell_boxes[cell, 0] - height <= centers[column + span] <= cell_boxes[cell, 2] + height
        ):
            span += 1
        key = (row_of[cell], column)
        if key in grid:
            grid[key] = (max(grid[key][0], span), grid[key][1] + " " + cell_texts[cell])
        else:
            grid[key] = (span, cell_texts[cell])

    out = ["<table>"]
    for row in range(len(rows)):
        tag = "th" if row == 0 and len(rows) > 1 else "td"
        cells_html, column = [], 0
        while column < len(columns):
            if (row, column) in grid:
                span, text = grid[(row, column)]
                attribute = f' colspan="{span}"' if span > 1 else ""
                cells_html.append(f"<{tag}{attribute}>{html.escape(text, quote=False)}</{tag}>")
                column += span
            else:
                cells_html.append(f"<{tag}></{tag}>")
                column += 1
        out.append("<tr>" + "".join(cells_html) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def page_blocks(boxes, texts, order, cls_logits, same):
    out = []
    for block in split_blocks(order, same):
        block_boxes = boxes[block, :4]
        out.append(
            {
                "label": BLOCK_CLASSES[int(cls_logits[block].sum(0).argmax())],
                "bbox": [
                    round(float(block_boxes[:, 0].min()), 1),
                    round(float(block_boxes[:, 1].min()), 1),
                    round(float(block_boxes[:, 2].max()), 1),
                    round(float(block_boxes[:, 3].max()), 1),
                ],
                "text": "\n".join(texts[index] for index in block),
            }
        )
    return out


def page_markdown(
    boxes,
    texts,
    order,
    cls_logits,
    same,
    keep_header=True,
    keep_footer=True,
    cells=None,
    tab=None,
    pictures=None,
):
    parts = []
    part_boxes = []
    for block in split_blocks(order, same):
        before = len(parts)
        _block_part(parts, block, boxes, texts, cls_logits, keep_header, keep_footer, cells, tab)
        if len(parts) > before:
            block_boxes = boxes[block, :4]
            part_boxes.append(
                (block_boxes[:, 0].min(), block_boxes[:, 1].min(), block_boxes[:, 2].max(), block_boxes[:, 3].max())
            )

    if pictures is not None and len(pictures):
        slots = {}
        for index, (x0, y0, x1, y1) in enumerate(np.asarray(pictures)[:, :4]):
            center_y = (y0 + y1) / 2
            at = next(
                (
                    slot
                    for slot, (bx0, by0, bx1, by1) in enumerate(part_boxes)
                    if (by0 + by1) / 2 > center_y and min(x1, bx1) > max(x0, bx0)
                ),
                len(parts),
            )
            slots.setdefault(at, []).append(f"![](picture-{index})")
        merged = []
        for index in range(len(parts) + 1):
            merged.extend(slots.get(index, []))
            if index < len(parts):
                merged.append(parts[index])
        parts = merged

    return "\n\n".join(part for part in parts if part)


def _block_part(parts, block, boxes, texts, cls_logits, keep_header, keep_footer, cells, tab):
    cls = int(cls_logits[block].sum(0).argmax())
    block_texts = [texts[index] for index in block]
    if cls == HEADER and not keep_header:
        return
    if cls == FOOTER and not keep_footer:
        return
    if all(_WHOLE_MATH.match(value.strip()) for value in block_texts):
        parts.append("\n".join("$$" + value.strip()[1:-1] + "$$" for value in block_texts))
        return
    if cls == SECTION:
        parts.append("## " + join_lines(block_texts))
    elif cls == TABLE and len(block) > 1 and tab is not None and heads_plausible(boxes[block, :4], tab[:, block][:, :, block]):
        parts.append(table_html_heads(boxes[block, :4], block_texts, tab[:, block][:, :, block]))
    elif cls == TABLE and len(block) > 1:
        cell_boxes, cell_texts = [], []
        for index in block:
            for box, text in (cells or {}).get(index) or [(boxes[index, :4], texts[index])]:
                cell_boxes.append(box)
                cell_texts.append(text)
        parts.append(table_html(np.array(cell_boxes, np.float32).reshape(-1, 4), cell_texts))
    elif cls == LIST:
        items = []
        for value in block_texts:
            if not items or _ITEM.match(value):
                items.append([value])
            else:
                items[-1].append(value)
        parts.append("\n".join(join_lines(item) for item in items))
    else:
        parts.append(join_lines(block_texts))


class _MarkdownTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self.rows[-1].append("".join(self._cell).strip())
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_markdown(value):
    parser = _MarkdownTableParser()
    parser.feed(value)
    rows = parser.rows
    if not rows:
        return value
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)
