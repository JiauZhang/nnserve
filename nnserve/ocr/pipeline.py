import itertools
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from nnm.models.indra_ocr import (
    IndraOCRDetection,
    IndraOCRMath,
    IndraOCROrder,
    IndraOCROrientation,
    IndraOCRRecognition,
    IndraOCRScript,
)
from nnm.models.indra_ocr.lm import NgramLM, beam_search
from nnm.models.indra_ocr.rtl import flip, has_rtl, is_rtl
from nnm.models.indra_ocr.vocab import Vocab

from nnserve.ocr import geometry, layout
from nnserve.ocr import order as reading

ADAPTIVE_MIN_H = 9.0
ADAPTIVE_TARGET_H = 12.0
ADAPTIVE_MAX_SIDE = 2048
SPLIT_GAP = 3.0
CELL_GAP = 1.8
GUTTER_GAP = 1.6
GUTTER_EDGES = 3
ORIENT_MIN = 0.95
VERTICAL = 4
MIN_CONF = 0.5
LM_GATE = 0.9
LM_ALPHA = 0.5
LM_BETA = 1.5
LM_BEAM = 8
LM_CAND_FLOOR = 3e-3
ROUTE_SURE = 0.95
GROUPS = ("base", "cjk", "rtl", "other")

LEADER_CHARS = set(".·…‥_-–—=")
DASHES = set("-–—")
BULLETS = set("•◦▪▫●○■□‣⁃∙*")
ENUMERATOR = re.compile(r"(\d{1,3}|[A-Za-z])[.)]|\((\d{1,3}|[A-Za-z]|[ivxlcIVXLC]{1,5})\)")
MARKER_GAP = 2.5


@dataclass
class Line:
    box: list[float]
    text: str
    conf: float


@dataclass
class Page:
    lines: list[Line] = field(default_factory=list)
    blocks: list[dict] = field(default_factory=list)
    markdown: str | None = None
    width: int = 0
    height: int = 0
    rotation: int = 0
    skew: float = 0.0
    vertical: bool = False

    @property
    def text(self):
        return "\n".join(line.text for line in self.lines)


class IndraOcrPipeline:
    def __init__(self, model_path, device="cpu", table_heads=True):
        self.path = Path(model_path)
        self.device = torch.device(device)
        self.table_heads = table_heads
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.long_side = self.meta["det_long_side"]
        self.height = self.meta["rec_height"]
        self.window = self.meta.get("rec_window", 1)
        self.script_groups = tuple(self.meta.get("script_groups", GROUPS))

        self.det = IndraOCRDetection.from_pretrained(self.path).to(self.device).eval()
        self.rec = IndraOCRRecognition.from_pretrained(self.path).to(self.device).eval()
        self.orient = IndraOCROrientation.from_pretrained(self.path).to(self.device).eval()
        self.script = IndraOCRScript.from_pretrained(self.path).to(self.device).eval()
        self.order = IndraOCROrder.from_pretrained(self.path).to(self.device).eval()
        self.math = IndraOCRMath.from_pretrained(self.path).to(self.device).eval()
        self.vocab = Vocab.from_json(self.meta["vocab"])

        self.groups = {}
        for group in self.script_groups:
            key = f"vocab_{group}"
            if group != "base" and key in self.meta:
                model = IndraOCRRecognition.from_pretrained(self.path, variant=group).to(self.device).eval()
                self.groups[group] = (model, Vocab.from_json(self.meta[key]))

        params = self.meta.get("lm_params", {})
        self.lms = {}
        base_lm = self.path / "lm.npz"
        if base_lm.exists():
            self.lms["base"] = (NgramLM(base_lm), *self._lm_params(params, "base"))
        for group in self.groups:
            lm_path = self.path / f"lm_{group}.npz"
            if lm_path.exists():
                self.lms[group] = (NgramLM(lm_path), *self._lm_params(params, group))

    @staticmethod
    def _lm_params(params, group):
        values = params.get(group, {})
        return (
            values.get("alpha", LM_ALPHA),
            values.get("beta", LM_BETA),
            values.get("gate", LM_GATE),
        )

    def _detect_at(self, gray, long_side):
        image, scale = geometry.page_to_input(gray, long_side)
        tensor = torch.from_numpy(image)[None, None].to(self.device)
        with torch.no_grad():
            out = self.det(tensor)[0].cpu().numpy()
        return geometry.decode_boxes(out), scale

    def detect(self, gray):
        boxes, scale = self._detect_at(gray, self.long_side)
        if len(boxes) >= 8:
            median_height = float(np.median(boxes[:, 3] - boxes[:, 1]))
            if median_height < ADAPTIVE_MIN_H:
                long_side = int(min(ADAPTIVE_MAX_SIDE, self.long_side * ADAPTIVE_TARGET_H / max(median_height, 1.0))) // 32 * 32
                if long_side > self.long_side:
                    boxes, scale = self._detect_at(gray, long_side)
        if len(boxes):
            boxes[:, :4] /= scale
            height, width = gray.shape
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        return boxes

    def _recognize_group(self, model, vocab, group, crops, tall, boxes, indices, batch, as_chars):
        out = {}
        for part in geometry.width_batches([crops[index].shape[1] for index in indices], batch):
            selected = [indices[index] for index in part]
            source = tall if group == "base" else crops
            array, widths = geometry.pad_batch([source[index] for index in selected])
            tensor = torch.from_numpy(array).to(self.device)
            width_tensor = torch.from_numpy(widths).to(self.device)
            with torch.no_grad():
                result = model.recognize(tensor, width_tensor)
            result = {key: value.cpu().numpy() for key, value in result.items()}

            stride = array.shape[-1] / max(result["ids"].shape[1], 1)
            for row, index in enumerate(selected):
                count = int(result["lengths"][row])
                sequence = result["ids"][row, :count]
                peak = result["conf"][row, :count]
                keep = (sequence != 0) & np.concatenate([[True], sequence[1:] != sequence[:-1]])
                x0, y0, x1, y1 = (float(value) for value in boxes[index, :4])
                box_height = max(1.0, y1 - y0)
                ix0 = np.floor(x0 - geometry.PAD_X * box_height)
                crop_width = np.ceil(x1 + geometry.PAD_X * box_height) - ix0
                factor = crop_width / crops[index].shape[1]
                positions = np.nonzero(keep)[0]
                ids = sequence[positions].tolist()
                confidence = float(peak[keep].mean()) if keep.any() else 0.0

                lm = self.lms.get(group)
                if lm is not None and MIN_CONF <= confidence < lm[3]:
                    ids, steps = beam_search(
                        result["blank"][row, :count],
                        result["topv"][row, :count],
                        result["topi"][row, :count],
                        lm[0],
                        lm[1],
                        lm[2],
                        LM_BEAM,
                        cand_floor=LM_CAND_FLOOR,
                    )
                    positions = np.array(steps, np.int64)

                xs = (ix0 + (positions + 0.5) * stride * factor).tolist()
                tokens = list(zip([vocab.chars[identifier - 1] for identifier in ids], xs)) if as_chars else list(zip(ids, xs))
                out[index] = (confidence, ids, tokens, vocab)

        return out

    def recognize(self, gray, boxes, batch=64, vertical=False):
        dark = [geometry.is_dark_line(gray, box) for box in boxes[:, :4]] if self.window > 1 else None
        crops = [
            geometry.crop_line(gray, box, self.height, dark=dark[index] if dark else None)
            for index, box in enumerate(boxes[:, :4])
        ]
        tall = (
            [
                geometry.crop_line(gray, box, self.height, window=self.window, dark=dark[index] if dark else None)
                for index, box in enumerate(boxes[:, :4])
            ]
            if self.window > 1
            else crops
        )

        best = [-1.0] * len(crops)
        out = [("", 0.0, [])] * len(crops)
        if vertical and "cjk" in self.groups:
            groups = [("cjk", *self.groups["cjk"], list(range(len(crops))), True)]
        else:
            names, sure = self.line_groups(crops, batch)
            base_indices = [index for index, name in enumerate(names) if name == "base" or not sure[index]]
            groups = [("base", self.rec, self.vocab, base_indices, False)]
            for group, (model, vocab) in self.groups.items():
                indices = [index for index, name in enumerate(names) if name == group]
                groups.append((group, model, vocab, indices, True))

        for group, model, vocab, indices, as_chars in groups:
            if not indices:
                continue
            results = self._recognize_group(model, vocab, group, crops, tall, boxes, indices, batch, as_chars)
            for index, (confidence, ids, tokens, token_vocab) in results.items():
                if confidence <= best[index]:
                    continue
                best[index] = confidence
                out[index] = (token_vocab.decode(ids), confidence, tokens) if confidence >= MIN_CONF else ("", confidence, [])

        if crops:
            array, widths = geometry.pad_batch(crops)
            with torch.no_grad():
                flags = self.math.flags(torch.from_numpy(array).to(self.device), torch.from_numpy(widths).to(self.device))
            math_rows = [index for index, flag in enumerate(flags) if bool(flag)]
        else:
            math_rows = []
        if math_rows:
            math_crops = [
                geometry.crop_line(gray, boxes[index, :4], self.math.config.encoder.input_height) for index in math_rows
            ]
            array, widths = geometry.pad_batch(math_crops)
            tensor = torch.from_numpy(array).to(self.device)
            width_tensor = torch.from_numpy(widths).to(self.device)
            with torch.no_grad():
                readings = self.math.decode(tensor, width_tensor)
            for index, (text, confidence) in zip(math_rows, readings):
                if "$" in text and confidence >= self.math.MIN_CONF:
                    center_x = float(boxes[index, 0] + boxes[index, 2]) / 2
                    out[index] = (text, 1.0, [(text, center_x)])
        return out

    def line_groups(self, crops, batch=64):
        names = ["base"] * len(crops)
        sure = [True] * len(crops)
        for indices in geometry.width_batches([crop.shape[1] for crop in crops], batch):
            array, widths = geometry.pad_batch([crops[index] for index in indices])
            tensor = torch.from_numpy(array).to(self.device)
            width_tensor = torch.from_numpy(widths).to(self.device)
            with torch.no_grad():
                probabilities = self.script(tensor, width_tensor).cpu().numpy()
            for row, index in enumerate(indices):
                group = self.script_groups[int(probabilities[row].argmax())]
                names[index] = group if group in self.groups else "base"
                sure[index] = float(probabilities[row].max()) > ROUTE_SURE
        return names, sure

    def _is_space(self, token):
        return token == " " or token == self.vocab.index.get(" ")

    def _decode(self, tokens):
        if tokens and isinstance(tokens[0][0], str):
            text = " ".join("".join(token[0] for token in tokens).split())
            if len(tokens[0]) == 2:
                text = unicodedata.normalize("NFC", text)
                if has_rtl(text):
                    text = flip(text, base_rtl=self._page_rtl or None)
            return text
        return self.vocab.decode([token[0] for token in tokens])

    def split_line(self, box, tokens, gap, at=None):
        box_height = max(1.0, float(box[3] - box[1]))
        segments = [[]]
        previous = None
        at = sorted(at or [])
        for token in tokens:
            x = token[1]
            if not self._is_space(token[0]):
                cut = previous is not None and (x - previous > gap * box_height or any(previous < value < x for value in at))
                if cut:
                    segments.append([])
                previous = x
            segments[-1].append(token)
        if len(segments) == 1:
            return [(list(box[:4]), self._decode(tokens), tokens)]

        out = []
        for segment in segments:
            text = self._decode(segment).strip()
            xs = [token[1] for token in segment if not self._is_space(token[0])]
            if text and xs:
                out.append(
                    (
                        [
                            max(box[0], min(xs) - 0.35 * box_height),
                            box[1],
                            min(box[2], max(xs) + 0.35 * box_height),
                            box[3],
                        ],
                        text,
                        segment,
                    )
                )
        return out

    def gutter_cuts(self, boxes, recognized):
        lefts, rights = boxes[:, 0], boxes[:, 2]
        cuts = []
        for index, (box, (_, _, tokens)) in enumerate(zip(boxes, recognized)):
            box_height = max(1.0, float(box[3] - box[1]))
            xs = [token[1] for token in tokens if not self._is_space(token[0])]
            found = []
            for left, right in itertools.pairwise(xs):
                if right - left <= GUTTER_GAP * box_height:
                    continue
                gap_left, gap_right = left + 0.35 * box_height, right - 0.35 * box_height
                near = (np.abs(rights - gap_left) < 0.5 * box_height) | (np.abs(lefts - gap_right) < 0.5 * box_height)
                near[index] = False
                if int(near.sum()) >= GUTTER_EDGES:
                    found.append((left + right) / 2)
            cuts.append(found)
        return cuts

    def reading_order(self, boxes, texts, page_wh):
        if len(boxes) < 2:
            return (reading.order_xycut(boxes[:, :4]) if len(boxes) else []), None
        features = torch.from_numpy(reading.line_features(boxes[:, :4], page_wh, texts)[None]).to(self.device)
        mask = torch.ones(1, len(boxes), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            order_logits, cls_logits, same_logits, table_logits = self.order(features, mask)
        order_logits = order_logits[0].cpu().numpy()
        cls_logits = cls_logits[0].cpu().numpy()
        same_logits = same_logits[0].cpu().numpy()
        table = table_logits[0].cpu().numpy() if self.table_heads else None
        order = reading.decode_blocks_order(order_logits, same_logits, boxes[:, :4], cls_logits)
        return order, (cls_logits, same_logits, table)

    def orientation(self, gray):
        tensor = torch.from_numpy(geometry.orient_input(gray))[None, None].to(self.device)
        with torch.no_grad():
            probabilities = self.orient(tensor)[0].cpu().numpy()
        index = int(probabilities.argmax())
        return index if probabilities[index] > ORIENT_MIN else 0

    def _turn(self, gray, turns, angle, small=None):
        quarter = 1 if turns == VERTICAL else -turns
        if turns == VERTICAL:
            gray = np.ascontiguousarray(np.rot90(gray, 1))
        elif turns:
            gray = np.ascontiguousarray(np.rot90(gray, -turns))
        if turns in (1, 3, VERTICAL):
            angle = geometry.estimate_skew(np.ascontiguousarray(np.rot90(small, quarter)) if small is not None else gray)
        if abs(angle) >= geometry.DESKEW_MIN:
            gray = geometry.rotate_page(gray, angle)
        else:
            angle = 0.0
        return gray, turns, angle

    def read(self, gray, turn=True, keep_header=True, keep_footer=True):
        small = geometry.preview(gray)
        angle = geometry.estimate_skew(small)
        if turn:
            oriented = geometry.rotate_page(small, angle) if abs(angle) >= geometry.DESKEW_MIN else small
            turns = self.orientation(oriented)
        else:
            turns = 0
        gray, turns, skew = self._turn(gray, turns, angle, small)
        vertical = turns == VERTICAL
        boxes = self.detect(gray)
        recognized = self.recognize(gray, boxes, vertical=vertical)
        page = self.assemble(
            (gray.shape[1], gray.shape[0]),
            boxes,
            recognized,
            vertical=vertical,
            keep_header=keep_header,
            keep_footer=keep_footer,
        )
        page.width, page.height = gray.shape[1], gray.shape[0]
        page.rotation, page.skew = (0 if vertical else 90 * turns), round(skew, 2)
        page.vertical = vertical
        if skew:
            size = (gray.shape[1], gray.shape[0])
            for line in page.lines:
                line.box = geometry.unrotate_box(line.box, skew, size)
            for block in page.blocks:
                block["bbox"] = geometry.unrotate_box(block["bbox"], skew, size)
        if vertical:
            width = gray.shape[0]

            def back(box):
                return [width - box[3], box[0], width - box[1], box[2]]

            for line in page.lines:
                line.box = back(line.box)
            for block in page.blocks:
                block["bbox"] = back(block["bbox"])
        return page

    def assemble(self, page_wh, boxes, recognized, vertical=False, keep_header=True, keep_footer=True):
        letters = [character for result in recognized for character in result[0] if character.isalpha()]
        rtl_count = sum(is_rtl(character) for character in letters)
        right_to_left = rtl_count > len(letters) - rtl_count
        self._page_rtl = right_to_left
        width = float(page_wh[0])

        def geometry_for(boxes_array):
            if not right_to_left:
                return boxes_array
            mirrored = np.array(boxes_array, np.float32).copy()
            mirrored[..., 0], mirrored[..., 2] = width - boxes_array[..., 2], width - boxes_array[..., 0]
            return mirrored

        new_boxes, new_recognized = [], []
        cuts = self.gutter_cuts(boxes, recognized) if len(boxes) else []
        for box, (text, confidence, tokens), at in zip(boxes, recognized, cuts):
            for new_box, new_text, new_tokens in self.split_line(box, tokens, SPLIT_GAP, at):
                new_boxes.append(new_box)
                new_recognized.append((new_text, confidence, new_tokens))

        boxes, recognized = tidy_lines(
            np.array(new_boxes, np.float32).reshape(-1, 4),
            new_recognized,
            right_to_left,
            self.vocab.index.get(" "),
        )
        texts = [item[0] for item in recognized]
        order, layout_info = self.reading_order(geometry_for(boxes), texts, page_wh)

        if vertical and layout_info is not None:
            blocks = reading.xycut_blocks(boxes[:, :4])
            order = [index for block in blocks for index in block]
            same = np.full((len(boxes), len(boxes)), -1.0, np.float32)
            for block in blocks:
                same[np.ix_(block, block)] = 1.0
            cls = layout_info[0].copy()
            for name in reading.GRID_LABELS:
                cls[:, reading.BLOCK_CLASSES.index(name)] = -1e4
            layout_info = (cls, same, None)
        elif layout_info is not None and layout_info[2] is not None:
            table_class = reading.BLOCK_CLASSES.index("Table")
            is_table = layout_info[0].argmax(-1) == table_class
            if is_table.any():
                new_boxes, new_recognized = [], []
                for index, (box, result) in enumerate(zip(boxes, recognized)):
                    parts = self.split_line(box, result[2], CELL_GAP) if is_table[index] else [(box, result[0], result[2])]
                    for new_box, new_text, new_tokens in parts:
                        new_boxes.append(new_box)
                        new_recognized.append((new_text, result[1], new_tokens))
                if len(new_boxes) != len(boxes):
                    boxes = np.array(new_boxes, np.float32).reshape(-1, 4)
                    recognized = new_recognized
                    texts = [item[0] for item in recognized]
                    order, layout_info = self.reading_order(geometry_for(boxes), texts, page_wh)

        lines = [
            Line([round(float(value), 1) for value in boxes[index, :4]], recognized[index][0], round(recognized[index][1], 4))
            for index in order
        ]
        markdown, blocks = None, []
        if layout_info is not None:
            blocks = layout.page_blocks(geometry_for(boxes[:, :4]), texts, order, layout_info[0], layout_info[1])
            for block in blocks:
                block["bbox"] = [round(float(value), 1) for value in geometry_for(np.array(block["bbox"]))]
            cells = {
                index: [
                    (list(geometry_for(np.array(box, np.float32))), text)
                    for box, text, _ in self.split_line(boxes[index], recognized[index][2], CELL_GAP)
                ]
                for index in range(len(boxes))
            }
            markdown = layout.page_markdown(
                geometry_for(boxes[:, :4]),
                texts,
                order,
                layout_info[0],
                layout_info[1],
                keep_header=keep_header,
                keep_footer=keep_footer,
                cells=cells,
                tab=layout_info[2],
            )
        elif len(boxes) < 2:
            markdown = "\n".join(texts)
        return Page(lines=lines, blocks=blocks, markdown=markdown)


def tidy_lines(boxes, recognized, right_to_left=False, space=None):
    keep = [True] * len(recognized)
    recognized = list(recognized)
    heights = boxes[:, 3] - boxes[:, 1]
    widths = boxes[:, 2] - boxes[:, 0]
    for index, (text, confidence, tokens) in enumerate(recognized):
        core = text.replace(" ", "")
        if core and set(core) <= LEADER_CHARS:
            wide = widths[index] >= 3 * heights[index]
            if len(core) >= 3 or wide or not set(core) & DASHES:
                keep[index] = False

    for index, (text, confidence, tokens) in enumerate(recognized):
        stripped = text.strip()
        if not keep[index] or not (stripped in BULLETS or ENUMERATOR.fullmatch(stripped)):
            continue
        best, best_gap = None, None
        for other in range(len(recognized)):
            if other == index or not keep[other] or not recognized[other][0].strip():
                continue
            overlap = min(boxes[index, 3], boxes[other, 3]) - max(boxes[index, 1], boxes[other, 1])
            if overlap < 0.5 * min(heights[index], heights[other]):
                continue
            gap = boxes[index, 0] - boxes[other, 2] if right_to_left else boxes[other, 0] - boxes[index, 2]
            in_gap = -0.3 * heights[index] <= gap <= MARKER_GAP * max(heights[index], heights[other])
            if in_gap and (best_gap is None or gap < best_gap):
                best, best_gap = other, gap
        if best is None:
            continue
        other_text, other_confidence, other_tokens = recognized[best]
        chars = bool(other_tokens) and isinstance(other_tokens[0][0], str)
        separator = " " if chars else space
        if not right_to_left and tokens and other_tokens and separator is not None and isinstance(tokens[0][0], str) == chars:
            x = float(boxes[index, 2] + boxes[best, 0]) / 2
            other_tokens = [*tokens, (separator, x, *tokens[-1][2:])] + list(other_tokens)
        recognized[best] = (f"{stripped} {other_text.strip()}", other_confidence, other_tokens)
        boxes[best] = [
            min(boxes[index, 0], boxes[best, 0]),
            min(boxes[index, 1], boxes[best, 1]),
            max(boxes[index, 2], boxes[best, 2]),
            max(boxes[index, 3], boxes[best, 3]),
        ]
        keep[index] = False

    indices = [index for index, value in enumerate(keep) if value]
    return boxes[indices].reshape(-1, 4), [recognized[index] for index in indices]
