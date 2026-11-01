import math

import cv2
import numpy as np

INPUT_MULTIPLE = 32
OUT_STRIDE = 2
LINE_HEIGHT = 32
PAD_X = 0.3
PAD_Y = 0.2
WIDTH_MULTIPLE = 8
ORIENT_SIDE = 768
PREVIEW_SIDE = 1600
DESKEW_MAX = 45.0
DESKEW_MIN = 0.3


def page_to_input(gray: np.ndarray, long_side: int):
    h, w = gray.shape
    scale = long_side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    if (nh, nw) != (h, w):
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        gray = cv2.resize(gray, (nw, nh), interpolation=interpolation)
    height = math.ceil(nh / INPUT_MULTIPLE) * INPUT_MULTIPLE
    width = math.ceil(nw / INPUT_MULTIPLE) * INPUT_MULTIPLE
    out = np.zeros((height, width), np.float32)
    out[:nh, :nw] = 1.0 - gray.astype(np.float32) / 255.0
    return out, scale


def decode_boxes(out: np.ndarray, threshold: float = 0.5, min_area: int = 2):
    prob = 1.0 / (1.0 + np.exp(-out[0]))
    mask = (prob > threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if count <= 1:
        return np.zeros((0, 5), np.float32)

    flat = labels.ravel()
    keep = flat > 0
    indices = flat[keep]
    ys = (np.indices(labels.shape)[0].ravel()[keep].astype(np.float32) + 0.5) * OUT_STRIDE
    top = ys - out[1].ravel()[keep]
    bottom = ys + out[2].ravel()[keep]
    score = prob.ravel()[keep]
    areas = np.bincount(indices, minlength=count).astype(np.float32)
    y0 = np.bincount(indices, weights=top, minlength=count) / np.maximum(areas, 1)
    y1 = np.bincount(indices, weights=bottom, minlength=count) / np.maximum(areas, 1)
    mean_score = np.bincount(indices, weights=score, minlength=count) / np.maximum(areas, 1)

    boxes = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < min_area:
            continue
        lh = max(1.0, y1[index] - y0[index])
        extend = min(0.15 * lh, 0.5 * lh)
        x0 = stats[index, cv2.CC_STAT_LEFT] * OUT_STRIDE - extend
        x1 = (stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH]) * OUT_STRIDE + extend
        boxes.append((x0, y0[index], x1, y1[index], mean_score[index]))
    return np.asarray(boxes, np.float32).reshape(-1, 5)


def is_dark_line(gray: np.ndarray, box) -> bool:
    height, width = gray.shape
    x0, y0, x1, y1 = (float(value) for value in box[:4])
    patch = gray[max(0, int(y0)) : min(height, int(y1) + 1), max(0, int(x0)) : min(width, int(x1) + 1)]
    if not patch.size:
        return False
    low, median, high = np.percentile(patch, (2, 50, 98))
    return bool(high - median > 2 * (median - low) + 8)


def crop_line(gray, box, height=LINE_HEIGHT, pad_y=PAD_Y, pad_x=PAD_X, max_width=4096, window=1, dark=None):
    pad_y = (window * (1 + 2 * pad_y) - 1) / 2
    height = height * window

    x0, y0, x1, y1 = (float(value) for value in box[:4])
    box_height = max(1.0, y1 - y0)
    left, right = x0 - pad_x * box_height, x1 + pad_x * box_height
    top, bottom = y0 - pad_y * box_height, y1 + pad_y * box_height
    page_height, page_width = gray.shape
    ix0, iy0 = math.floor(left), math.floor(top)
    ix1, iy1 = math.ceil(right), math.ceil(bottom)
    crop_width, crop_height = max(1, ix1 - ix0), max(1, iy1 - iy0)
    sx0, sy0 = max(0, ix0), max(0, iy0)
    sx1, sy1 = min(page_width, ix1), min(page_height, iy1)

    if dark is None:
        dark = is_dark_line(gray, box)
    crop = np.full((crop_height, crop_width), 0 if dark else 255, np.uint8)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - iy0 : sy1 - iy0, sx0 - ix0 : sx1 - ix0] = gray[sy0:sy1, sx0:sx1]

    out_width = max(WIDTH_MULTIPLE, min(max_width, round(crop_width * height / crop_height)))
    interpolation = cv2.INTER_AREA if crop_height > height else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (out_width, height), interpolation=interpolation)
    ink = resized.astype(np.float32) / 255.0
    return ink if dark else 1.0 - ink


def pad_batch(lines, multiple=WIDTH_MULTIPLE):
    height = lines[0].shape[0]
    width = max(line.shape[1] for line in lines)
    width = int(math.ceil(width / multiple) * multiple)
    out = np.zeros((len(lines), 1, height, width), np.float32)
    widths = np.zeros(len(lines), np.int64)
    for index, line in enumerate(lines):
        out[index, 0, :, : line.shape[1]] = line
        widths[index] = line.shape[1]
    return out, widths


def width_batches(widths, max_batch=64, max_ratio=1.3, min_width=64, max_area=64 * 640):
    order = sorted(range(len(widths)), key=lambda index: widths[index])
    batches = []
    for index in order:
        if (
            batches
            and len(batches[-1]) < max_batch
            and max(widths[index], min_width) <= max_ratio * max(widths[batches[-1][0]], min_width)
            and (len(batches[-1]) + 1) * max(widths[index], min_width) <= max_area
        ):
            batches[-1].append(index)
        else:
            batches.append([index])
    return batches


def preview(gray: np.ndarray, side=PREVIEW_SIDE):
    height, width = gray.shape
    scale = side / max(height, width)
    if scale >= 1:
        return gray
    return cv2.resize(
        gray,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def orient_input(gray: np.ndarray, side=ORIENT_SIDE):
    height, width = gray.shape
    scale = side / max(height, width)
    small = cv2.resize(
        gray,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    out = np.zeros((side, side), np.float32)
    out[: small.shape[0], : small.shape[1]] = 1.0 - small.astype(np.float32) / 255.0
    return out


def rotate_page(gray: np.ndarray, degrees: float):
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def unrotate_box(box, degrees: float, size):
    width, height = size
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -degrees, 1.0)
    x0, y0, x1, y1 = box[:4]
    corners = matrix @ np.array([[x0, x1, x0, x1], [y0, y0, y1, y1], [1, 1, 1, 1]], float)
    return [
        round(float(np.clip(corners[0].min(), 0, width)), 1),
        round(float(np.clip(corners[1].min(), 0, height)), 1),
        round(float(np.clip(corners[0].max(), 0, width)), 1),
        round(float(np.clip(corners[1].max(), 0, height)), 1),
    ]


def _ink_thumb(gray, side):
    height, width = gray.shape
    scale = side / max(height, width)
    small = cv2.resize(
        gray,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return 255 - small


def _row_sharpness(ink, angle):
    size = (ink.shape[1], ink.shape[0])
    rotated = cv2.warpAffine(
        ink,
        cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0),
        size,
    )
    profile = rotated.sum(1, dtype=np.int64).astype(np.float64)
    return float(((profile[1:] - profile[:-1]) ** 2).sum())


def estimate_skew(gray, side=800):
    fine = _ink_thumb(gray, side)
    if fine.mean() < 0.003 * 255:
        return 0.0
    coarse = _ink_thumb(gray, side // 2)
    grid = np.arange(-DESKEW_MAX, DESKEW_MAX + 1e-6, 2.0)
    scores = [_row_sharpness(coarse, angle) for angle in grid]
    starts = {0.0, *(float(grid[index]) for index in np.argsort(scores)[-2:])}
    near = [angle for start in starts for angle in np.arange(start - 2.0, start + 2.0 + 1e-6, 0.5)]
    seen = dict.fromkeys(near)
    seen.update(zip(seen, [_row_sharpness(fine, angle) for angle in list(seen)]))
    best = max(near, key=seen.__getitem__)
    last = np.arange(best - 0.5, best + 0.5 + 1e-6, 0.1)
    values = [_row_sharpness(fine, angle) for angle in last]
    return float(max(zip(last, values), key=lambda pair: pair[1])[0])
