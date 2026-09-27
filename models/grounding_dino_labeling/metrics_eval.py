"""
Module tính toán và đánh giá Metrics chuẩn COCO & Operating Metrics (IoU 0.50):
- Precision
- Recall
- AP50
- AP75
- mAP50:95
- Mean IoU
- Throughput (FPS) / Latency (ms)
"""

from __future__ import annotations
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def load_yolo_bboxes(txt_path: str | Path, img_width: int, img_height: int) -> np.ndarray:
    """Đọc file nhãn YOLO txt (format: class cx cy w h chuẩn hoá) -> trả về boxes_xyxy (pixel)."""
    p = Path(txt_path)
    if not p.exists() or p.stat().st_size == 0:
        return np.empty((0, 4), dtype=np.float32)

    boxes = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # class cx cy nw nh
                _, cx, cy, nw, nh = map(float, parts[:5])
                bw = nw * img_width
                bh = nh * img_height
                x1 = (cx * img_width) - (bw / 2.0)
                y1 = (cy * img_height) - (bh / 2.0)
                x2 = x1 + bw
                y2 = y1 + bh
                boxes.append([x1, y1, x2, y2])

    return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)


def box_iou_numpy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Tính ma trận IoU giữa 2 tập box [[x1, y1, x2, y2], ...]."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, a_min=0, a_max=None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area_a[:, None] + area_b[None, :] - inter
    union = np.maximum(union, 1e-9)

    return inter / union


def evaluate_single_image_operating(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thresh: float = 0.50
) -> dict[str, Any]:
    """Tính TP, FP, FN, Precision, Recall, Mean IoU cho 1 ảnh đơn lẻ tại ngưỡng IoU cho trước."""
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)

    if num_pred == 0 and num_gt == 0:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "mean_iou": 1.0, "matched_ious": []
        }
    if num_pred == 0:
        return {
            "tp": 0, "fp": 0, "fn": num_gt,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "mean_iou": 0.0, "matched_ious": []
        }
    if num_gt == 0:
        return {
            "tp": 0, "fp": num_pred, "fn": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "mean_iou": 0.0, "matched_ious": []
        }

    iou_mat = box_iou_numpy(pred_boxes, gt_boxes)
    sort_idx = np.argsort(pred_scores)[::-1]

    used_gt = set()
    tp = 0
    matched_ious = []

    for p_i in sort_idx:
        # Tìm GT khớp tốt nhất chưa được gán
        best_g = None
        best_iou = -1.0
        for g_i in range(num_gt):
            if g_i not in used_gt:
                if iou_mat[p_i, g_i] > best_iou:
                    best_iou = float(iou_mat[p_i, g_i])
                    best_g = g_i

        if best_g is not None and best_iou >= iou_thresh:
            used_gt.add(best_g)
            tp += 1
            matched_ious.append(best_iou)

    fp = num_pred - tp
    fn = num_gt - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_iou": mean_iou,
        "matched_ious": matched_ious
    }


def evaluate_dataset_coco(records: list[dict[str, Any]]) -> dict[str, float]:
    """Tính các chỉ số chuẩn COCO (mAP@[0.50:0.95], AP50, AP75, AR) bằng pycocotools."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return {
            "mAP50_95": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AR": 0.0
        }

    coco_gt_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "window"}]
    }
    coco_dt_rows = []
    ann_id = 1

    has_gt = False
    for rec in records:
        img_id = int(rec["image_id"])
        coco_gt_dict["images"].append({
            "id": img_id,
            "file_name": rec.get("file_name", f"image_{img_id}.jpg"),
            "width": int(rec["width"]),
            "height": int(rec["height"])
        })

        gt_boxes = rec.get("gt_boxes", [])
        if len(gt_boxes) > 0:
            has_gt = True
            for box in gt_boxes:
                x1, y1, x2, y2 = map(float, box)
                bw = x2 - x1
                bh = y2 - y1
                coco_gt_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0
                })
                ann_id += 1

        pred_boxes = rec.get("pred_boxes", [])
        pred_scores = rec.get("pred_scores", [])
        for box, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = map(float, box)
            bw = x2 - x1
            bh = y2 - y1
            coco_dt_rows.append({
                "image_id": img_id,
                "category_id": 1,
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "score": float(score)
            })

    if not has_gt or len(coco_dt_rows) == 0:
        return {
            "mAP50_95": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AR": 0.0
        }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(coco_gt_dict, f)
        temp_gt_path = f.name

    try:
        coco_gt = COCO(temp_gt_path)
        coco_dt = coco_gt.loadRes(coco_dt_rows)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

        return {
            "mAP50_95": float(ev.stats[0]),
            "AP50": float(ev.stats[1]),
            "AP75": float(ev.stats[2]),
            "AR": float(ev.stats[8])
        }
    except Exception as e:
        print(f"Lỗi khi chạy COCOeval: {e}")
        return {
            "mAP50_95": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AR": 0.0
        }
    finally:
        Path(temp_gt_path).unlink(missing_ok=True)


def draw_predictions_and_gt(
    image: Image.Image,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray | None = None,
    show_gt: bool = True
) -> Image.Image:
    """Vẽ bounding box dự đoán (Màu Đỏ/Cam) và Ground Truth (Màu Xanh Cyan/Lam) lên ảnh."""
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    w, h = vis.size

    # 1. Vẽ Ground Truth (Màu Xanh Dương/Cyan)
    if show_gt and gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            x1, y1, x2, y2 = map(float, box)
            draw.rectangle([x1, y1, x2, y2], outline="#00E5FF", width=3)
            # Badge nhỏ cho GT
            gt_text = "GT"
            badge_y = max(0.0, y1 - 13.0)
            draw.rectangle([x1, badge_y, x1 + 24, badge_y + 12], fill="#00E5FF")
            draw.text((x1 + 2, badge_y - 1), gt_text, fill="black")

    # 2. Vẽ Predictions (Màu Đỏ / Cam rực rỡ)
    if len(pred_boxes) > 0:
        for box, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = map(float, box)
            draw.rectangle([x1, y1, x2, y2], outline="#FF2A55", width=3)
            # Badge
            text = f"win {float(score):.2f}"
            badge_y = min(h - 14, max(0.0, y1 - 15.0))
            badge_w = len(text) * 7.0 + 4
            draw.rectangle([x1, badge_y, x1 + badge_w, badge_y + 14], fill="#FF2A55")
            draw.text((x1 + 2, badge_y), text, fill="white")

    # 3. Vẽ chú thích (Legend) góc trên bên trái
    legend_bg = [8, 8, 230, 40 if (gt_boxes is not None and len(gt_boxes) > 0) else 26]
    draw.rectangle(legend_bg, fill="#000000AA", outline="#FFFFFF")
    draw.rectangle([14, 14, 26, 22], fill="#FF2A55")
    draw.text((32, 11), f"Pred ({len(pred_boxes)} windows)", fill="#FFFFFF")
    if show_gt and gt_boxes is not None and len(gt_boxes) > 0:
        draw.rectangle([14, 26, 26, 34], fill="#00E5FF")
        draw.text((32, 23), f"Ground Truth ({len(gt_boxes)} windows)", fill="#FFFFFF")

    return vis
