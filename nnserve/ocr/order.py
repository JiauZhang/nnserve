import math
import unicodedata

import numpy as np

N_GEOM = 10
N_TEXT = 6
N_FEATS = N_GEOM + N_TEXT
SENTENCE_END = ".!?:;。！？：；．"

BLOCK_CLASSES = [
    "Text",
    "Section-Header",
    "Page-Header",
    "Page-Footer",
    "List-Group",
    "Table",
    "Caption",
    "Footnote",
    "Table-Of-Contents",
    "Form",
]
_CLASS_ID = {name: index for index, name in enumerate(BLOCK_CLASSES)}
GRID_LABELS = {"Table", "Form", "Table-Of-Contents"}


def display_len(text):
    return sum(2 if unicodedata.east_asian_width(character) in "WF" else 1 for character in text)


def line_features(boxes, page_wh, texts=None):
    width, height = page_wh
    boxes = boxes.astype(np.float32)
    count = len(boxes)
    if count == 0:
        return np.zeros((0, N_FEATS), np.float32)

    box_width = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-3)
    box_height = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-3)
    median_height = float(np.median(box_height))
    geometry = np.stack(
        [
            boxes[:, 0] / width,
            boxes[:, 1] / height,
            boxes[:, 2] / width,
            boxes[:, 3] / height,
            (boxes[:, 0] + boxes[:, 2]) / (2 * width),
            (boxes[:, 1] + boxes[:, 3]) / (2 * height),
            box_width / width,
            np.log(box_height / median_height),
            np.log(box_width / box_height) / 4.0,
            np.full(count, median_height / height * 20.0),
        ],
        1,
    )

    text = np.zeros((count, N_TEXT), np.float32)
    if texts is not None:
        for index, value in enumerate(texts):
            value = value.strip()
            if not value:
                continue
            text[index, 0] = math.log1p(display_len(value)) / 5.0
            text[index, 1] = sum(character.isdigit() for character in value) / len(value)
            text[index, 2] = float(value[0] in "•-–▪◦*·→" or (value[:3].rstrip(".)").isdigit() and len(value) > 3))
            text[index, 3] = float(value.endswith("-"))
            text[index, 4] = float(value[-1] in SENTENCE_END)
            text[index, 5] = float(value[0].isupper())
    return np.concatenate([geometry, text], 1).astype(np.float32)


def decode_order(logits, sweeps=4):
    count = logits.shape[0]
    if count <= 1:
        return list(range(count))
    log_prob = -np.logaddexp(0.0, -logits)
    order = np.argsort(-log_prob.sum(1) + log_prob.sum(0))
    for _ in range(sweeps):
        improved = False
        for item in order.copy():
            rest = order[order != item]
            before = log_prob[rest, item]
            after = log_prob[item, rest]
            gains = np.concatenate(([0.0], np.cumsum(before))) + np.concatenate((np.cumsum(after[::-1])[::-1], [0.0]))
            best = int(np.argmax(gains))
            current = int(np.nonzero(order == item)[0][0])
            if gains[best] > gains[current] + 1e-6:
                order = np.insert(rest, best, item)
                improved = True
        if not improved:
            break
    return [int(value) for value in order]


def order_top_down(boxes):
    if len(boxes) == 0:
        return []
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2
    height = np.median(boxes[:, 3] - boxes[:, 1])
    return sorted(
        range(len(boxes)),
        key=lambda index: (round(center_y[index] / max(height * 0.5, 1e-3)), boxes[index, 0]),
    )


def order_xycut(boxes):
    def best_cut(part, axis):
        low, high = (part[:, 1], part[:, 3]) if axis == 1 else (part[:, 0], part[:, 2])
        order = np.argsort(low)
        end = high[order[0]]
        best_gap, best_index = 0.0, -1
        for index in range(1, len(order)):
            gap = low[order[index]] - end
            if gap > best_gap:
                best_gap, best_index = gap, index
            end = max(end, high[order[index]])
        return best_gap, best_index, order

    def walk(ids):
        if len(ids) <= 1:
            return list(ids)
        part = boxes[ids]
        gap_y, cut_y, order_y = best_cut(part, 1)
        gap_x, cut_x, order_x = best_cut(part, 0)
        if cut_y < 0 and cut_x < 0:
            return [ids[index] for index in order_top_down(part)]
        cut, order = (cut_y, order_y) if gap_y >= gap_x else (cut_x, order_x)
        return walk([ids[index] for index in order[:cut]]) + walk([ids[index] for index in order[cut:]])

    return walk(list(range(len(boxes))))


def xycut_blocks(boxes, min_gap=1.0):
    if len(boxes) == 0:
        return []
    height = float(np.median(boxes[:, 3] - boxes[:, 1]))

    def best_cut(part, axis):
        low, high = (part[:, 1], part[:, 3]) if axis == 1 else (part[:, 0], part[:, 2])
        order = np.argsort(low)
        end = high[order[0]]
        best_gap, best_index = 0.0, -1
        for index in range(1, len(order)):
            gap = low[order[index]] - end
            if gap > best_gap:
                best_gap, best_index = gap, index
            end = max(end, high[order[index]])
        return best_gap, best_index, order

    def walk(ids):
        part = boxes[ids]
        if len(ids) > 1:
            gap_y, cut_y, order_y = best_cut(part, 1)
            gap_x, cut_x, order_x = best_cut(part, 0)
            gap, cut, order = (gap_y, cut_y, order_y) if gap_y >= gap_x else (gap_x, cut_x, order_x)
            if cut >= 0 and gap >= min_gap * height:
                return walk([ids[index] for index in order[:cut]]) + walk([ids[index] for index in order[cut:]])
        return [[ids[index] for index in order_xycut(part)]]

    return walk(list(range(len(boxes))))


def _cluster(logits):
    count = logits.shape[0]
    if count == 0:
        return []
    scores = logits.astype(np.float64).copy()
    np.fill_diagonal(scores, 0.0)
    size = np.ones(count)
    alive = np.ones(count, bool)
    members = [[index] for index in range(count)]
    means = scores.copy()
    np.fill_diagonal(means, -np.inf)
    best = means.argmax(1)
    best_value = means[np.arange(count), best]

    for _ in range(count - 1):
        left = int(np.argmax(best_value))
        right = int(best[left])
        if best_value[left] <= 0:
            break
        left, right = min(left, right), max(left, right)
        scores[left] += scores[right]
        scores[:, left] += scores[:, right]
        scores[left, left] = 0.0
        size[left] += size[right]
        alive[right] = False
        members[left] += members[right]
        row = scores[left] / (size[left] * size)
        row[~alive] = -np.inf
        row[left] = -np.inf
        means[left], means[:, left] = row, row
        means[right], means[:, right] = -np.inf, -np.inf
        best_value[right] = -np.inf
        best[left] = int(row.argmax())
        best_value[left] = row[best[left]]
        stale = alive & ((best == left) | (best == right))
        stale[left] = False
        for index in np.flatnonzero(stale):
            best[index] = int(means[index].argmax())
            best_value[index] = means[index, best[index]]
        updated = alive & (row > best_value)
        updated[left] = False
        best[updated], best_value[updated] = left, row[updated]

    return [sorted(members[index]) for index in range(count) if alive[index]]


def decode_blocks_order(order_logits, same_logits, boxes, cls_logits=None):
    count = order_logits.shape[0]
    if count <= 1:
        return list(range(count))
    blocks = _cluster(same_logits)
    grid = {_CLASS_ID[name] for name in GRID_LABELS if name in _CLASS_ID}
    inner = []
    for block in blocks:
        block_boxes = boxes[block, :4]
        if cls_logits is not None and int(cls_logits[block].sum(0).argmax()) in grid:
            local = order_top_down(block_boxes)
        else:
            local = order_xycut(block_boxes)
        inner.append([block[index] for index in local])

    block_count = len(blocks)
    membership = np.zeros((block_count, count), np.float64)
    for index, block in enumerate(blocks):
        membership[index, block] = 1.0 / len(block)
    upper = np.triu(membership @ order_logits.astype(np.float64) @ membership.T, 1)
    between = (upper - upper.T).astype(np.float32)
    return [index for block in decode_order(between) for index in inner[block]]
