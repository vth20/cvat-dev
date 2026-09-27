"""
Script Inference Cửa Sổ (Window Detection) bằng Grounding DINO
Tự động ưu tiên tải model từ thư mục offline local `saved_model`.
Hỗ trợ:
- Chạy inference cho 1 ảnh lẻ hoặc cả thư mục ảnh
- Tự động vẽ bounding box kết quả và lưu ra thư mục output
- Xuất nhãn bbox ra file JSON / TXT
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

# Tắt cảnh báo FutureWarning từ thư viện transformers
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import batched_nms, box_convert
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Cấu hình mặc định
DEFAULT_PROMPT = "window."
DEFAULT_BOX_THRESHOLD = 0.22
DEFAULT_TEXT_THRESHOLD = 0.20
DEFAULT_NMS_IOU = 0.45
DEFAULT_MIN_BOX_AREA = 4.0
DEFAULT_MAX_BOXES = 900

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_detector(model_path_or_id: str | Path, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Nạp processor và model từ thư mục local hoặc Hugging Face Hub."""
    print(f"📦 Đang nạp Grounding DINO từ: {model_path_or_id} (Thiết bị: {device.upper()})...")
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_path_str = str(model_path_or_id)
    try:
        processor = AutoProcessor.from_pretrained(model_path_str)
    except (TypeError, Exception):
        from transformers import GroundingDinoProcessor, GroundingDinoImageProcessor, AutoTokenizer
        img_proc = GroundingDinoImageProcessor.from_pretrained(model_path_str)
        tokenizer = AutoTokenizer.from_pretrained(model_path_str)
        processor = GroundingDinoProcessor(image_processor=img_proc, tokenizer=tokenizer)

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_path_str,
        torch_dtype=dtype,
    ).to(device).eval()

    print("✓ Nạp mô hình thành công!")
    return processor, model, dtype, device


def predict_single_image(
    image: Image.Image,
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    prompt: str = DEFAULT_PROMPT,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    nms_iou: float = DEFAULT_NMS_IOU,
    min_box_area: float = DEFAULT_MIN_BOX_AREA,
    max_boxes: int = DEFAULT_MAX_BOXES,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> dict[str, Any]:
    """Inference một bức ảnh và trả về danh sách box, score, label."""
    image = image.convert("RGB")
    width, height = image.size

    t0 = time.perf_counter()
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast("cuda", dtype=dtype):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=[(height, width)],
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=[(height, width)],
        )[0]

    boxes = results["boxes"].float().cpu().numpy()
    scores = results["scores"].float().cpu().numpy()
    # Ưu tiên lấy text_labels (chuẩn mới) hoặc labels (chuẩn cũ)
    labels = results.get("text_labels", results.get("labels", ["window"] * len(boxes)))

    # Lọc bỏ các box có diện tích quá nhỏ
    if len(boxes) > 0:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        valid_area = areas >= min_box_area
        boxes = boxes[valid_area]
        scores = scores[valid_area]
        labels = [labels[i] for i in range(len(labels)) if valid_area[i]]

    # Áp dụng NMS
    if len(boxes) > 0:
        cids = torch.zeros(len(boxes), dtype=torch.int64)
        keep = batched_nms(
            torch.from_numpy(boxes),
            torch.from_numpy(scores),
            cids,
            float(nms_iou),
        ).cpu().numpy()[:max_boxes]

        boxes = boxes[keep]
        scores = scores[keep]
        labels = [labels[k] for k in keep]

    total_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "width": width,
        "height": height,
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "latency_ms": total_ms,
    }


def draw_predictions(image: Image.Image, pred: dict[str, Any], outline_color: str = "#FF0000") -> Image.Image:
    """Vẽ bounding box lên ảnh."""
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)

    for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
        x1, y1, x2, y2 = map(float, box)
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=3)
        # Vẽ badge
        text = f"{label} {float(score):.2f}"
        badge_y = max(0.0, y1 - 15.0)
        draw.rectangle([x1, badge_y, x1 + len(text) * 7.5, badge_y + 14], fill=outline_color)
        draw.text((x1 + 2, badge_y), text, fill="white")

    return vis


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Window Detection bằng Grounding DINO")
    parser.add_argument("--input", "-i", type=str, required=True, help="Đường dẫn đến 1 file ảnh hoặc thư mục chứa ảnh")
    parser.add_argument("--output", "-o", type=str, default="./inference_outputs", help="Thư mục lưu kết quả trực quan và json")
    parser.add_argument("--model-path", "-m", type=str, default=None, help="Đường dẫn model local (mặc định ưu tiên './saved_model')")
    parser.add_argument("--prompt", "-p", type=str, default=DEFAULT_PROMPT, help=f"Text prompt (mặc định: '{DEFAULT_PROMPT}')")
    parser.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD, help=f"Ngưỡng confidence box (mặc định: {DEFAULT_BOX_THRESHOLD})")
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD, help=f"Ngưỡng text confidence (mặc định: {DEFAULT_TEXT_THRESHOLD})")
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU, help=f"Ngưỡng NMS IoU (mặc định: {DEFAULT_NMS_IOU})")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị: 'cuda' hoặc 'cpu'")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Xác định đường dẫn model
    if args.model_path:
        model_target = Path(args.model_path)
    else:
        local_saved = Path(__file__).resolve().parent / "saved_model"
        if local_saved.exists() and (local_saved / "config.json").exists():
            model_target = local_saved
        else:
            model_target = "IDEA-Research/grounding-dino-base"

    # Nạp mô hình
    processor, model, dtype, device = load_detector(model_target, device=args.device)

    # Thu thập danh sách ảnh
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    else:
        raise FileNotFoundError(f"Không tìm thấy đường dẫn đầu vào: {input_path}")

    if not image_files:
        print(f"⚠️ Không tìm thấy file ảnh hợp lệ nào trong {input_path}")
        return

    print(f"\n🚀 Bắt đầu inference trên {len(image_files)} ảnh...")
    total_boxes_all = 0
    times = []

    for idx, img_path in enumerate(image_files, 1):
        image = Image.open(img_path).convert("RGB")
        res = predict_single_image(
            image=image,
            processor=processor,
            model=model,
            prompt=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            nms_iou=args.nms_iou,
            dtype=dtype,
            device=device,
        )

        total_boxes_all += len(res["boxes"])
        times.append(res["latency_ms"])

        # Lưu ảnh visual
        vis_img = draw_predictions(image, res)
        out_vis_path = output_dir / f"{img_path.stem}_pred.jpg"
        vis_img.save(out_vis_path, quality=95)

        # Lưu JSON dự đoán
        json_out = {
            "file_name": img_path.name,
            "width": res["width"],
            "height": res["height"],
            "latency_ms": res["latency_ms"],
            "boxes_xyxy": res["boxes"].tolist(),
            "scores": res["scores"].tolist(),
            "labels": res["labels"],
        }
        (output_dir / f"{img_path.stem}_pred.json").write_text(json.dumps(json_out, indent=2), encoding="utf-8")

        print(f"[{idx}/{len(image_files)}] {img_path.name} -> {len(res['boxes'])} windows ({res['latency_ms']:.1f} ms)")

    avg_time = np.mean(times)
    fps = 1000.0 / avg_time if avg_time > 0 else 0.0
    print("\n" + "=" * 70)
    print(f"✓ HOÀN THÀNH XỬ LÝ {len(image_files)} ẢNH!")
    print(f"• Tổng số cửa sổ phát hiện: {total_boxes_all:,}")
    print(f"• Thời gian trung bình: {avg_time:.1f} ms/ảnh (~{fps:.1f} FPS)")
    print(f"• Toàn bộ kết quả (ảnh vẽ box + JSON) đã lưu tại: {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
