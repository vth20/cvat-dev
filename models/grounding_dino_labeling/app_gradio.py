import os
import sys
import time
import json
import zipfile
import uuid
from pathlib import Path
import gradio as gr
from PIL import Image
import torch
import numpy as np
import pandas as pd

# Import core inference & evaluation logic
from inference import (
    load_detector,
    predict_single_image,
    DEFAULT_PROMPT,
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    DEFAULT_NMS_IOU,
    DEFAULT_MAX_BOXES
)
from metrics_eval import (
    load_yolo_bboxes,
    evaluate_single_image_operating,
    evaluate_dataset_coco,
    draw_predictions_and_gt
)

MAIN_DIR = Path(__file__).resolve().parent
LOCAL_MODEL = MAIN_DIR / "saved_model"
MODEL_PATH = LOCAL_MODEL if (LOCAL_MODEL.exists() and (LOCAL_MODEL / "config.json").exists()) else "IDEA-Research/grounding-dino-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEMO_DIR = MAIN_DIR / "Window_detection_a.yolov11" / "train"
DEMO_IMG_DIR = DEMO_DIR / "images"
DEMO_LBL_DIR = DEMO_DIR / "labels"

print("=" * 70)
print(f"🚀 Đang khởi động Gradio App...")
print(f"📁 Model Path: {MODEL_PATH}")
print(f"💻 Device: {DEVICE.upper()} ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'})")
print("=" * 70)

PROCESSOR, MODEL, DTYPE, DEVICE = load_detector(MODEL_PATH, device=DEVICE)

EXPORT_ROOT = MAIN_DIR / "gradio_exports"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)


def get_demo_image_paths():
    if DEMO_IMG_DIR.exists():
        images = sorted(
            list(DEMO_IMG_DIR.glob("*.jpg")) +
            list(DEMO_IMG_DIR.glob("*.png")) +
            list(DEMO_IMG_DIR.glob("*.jpeg"))
        )
        return [str(p) for p in images[:5]]
    return []


def run_pipeline(
    files,
    progress=gr.Progress(track_tqdm=True)
):
    empty_df = pd.DataFrame(columns=["Tên ảnh", "GT", "Pred", "Precision", "Recall", "Mean IoU", "Latency (ms)"])
    if not files:
        return [], None, None, empty_df, "⚠️ Vui lòng tải lên ít nhất 1 ảnh hoặc nhấn **Nạp ảnh mẫu Demo**!", "-", "-", "-", "-", "-", "-", "-"

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = EXPORT_ROOT / run_id

    yolo_dir = run_dir / "yolo"
    yolo_img = yolo_dir / "images"
    yolo_lbl = yolo_dir / "labels"
    yolo_img.mkdir(parents=True, exist_ok=True)
    yolo_lbl.mkdir(parents=True, exist_ok=True)

    coco_dir = run_dir / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "window"}]
    }

    gallery_results = []
    records = []
    table_rows = []

    total_tp, total_fp, total_fn = 0, 0, 0
    all_matched_ious = []
    times = []
    ann_id_counter = 1
    total_gt_count = 0
    total_pred_count = 0

    total_files = len(files)
    for idx, file_path_str in enumerate(files, 1):
        progress((idx - 1) / total_files, desc=f"Đang phân tích {idx}/{total_files}...")

        path = Path(file_path_str)
        pil_img = Image.open(path).convert("RGB")
        w, h = pil_img.size

        gt_boxes = np.empty((0, 4), dtype=np.float32)
        demo_lbl_candidate = DEMO_LBL_DIR / f"{path.stem}.txt"
        if demo_lbl_candidate.exists():
            gt_boxes = load_yolo_bboxes(demo_lbl_candidate, w, h)
        else:
            local_lbl_candidate = path.parent / f"{path.stem}.txt"
            if local_lbl_candidate.exists():
                gt_boxes = load_yolo_bboxes(local_lbl_candidate, w, h)

        total_gt_count += len(gt_boxes)

        res = predict_single_image(
            image=pil_img,
            processor=PROCESSOR,
            model=MODEL,
            prompt=DEFAULT_PROMPT,
            box_threshold=float(DEFAULT_BOX_THRESHOLD),
            text_threshold=float(DEFAULT_TEXT_THRESHOLD),
            nms_iou=float(DEFAULT_NMS_IOU),
            max_boxes=int(DEFAULT_MAX_BOXES),
            dtype=DTYPE,
            device=DEVICE
        )

        num_preds = len(res["boxes"])
        total_pred_count += num_preds
        times.append(res["latency_ms"])

        op_metric = evaluate_single_image_operating(res["boxes"], res["scores"], gt_boxes, iou_thresh=0.50)
        total_tp += op_metric["tp"]
        total_fp += op_metric["fp"]
        total_fn += op_metric["fn"]
        all_matched_ious.extend(op_metric["matched_ious"])

        records.append({
            "image_id": idx,
            "file_name": path.name,
            "width": w,
            "height": h,
            "pred_boxes": res["boxes"].tolist(),
            "pred_scores": res["scores"].tolist(),
            "gt_boxes": gt_boxes.tolist()
        })

        gt_str = str(len(gt_boxes)) if len(gt_boxes) > 0 else "N/A"
        prec_str = f"{op_metric['precision']:.1%}" if len(gt_boxes) > 0 else "N/A"
        rec_str = f"{op_metric['recall']:.1%}" if len(gt_boxes) > 0 else "N/A"
        iou_str = f"{op_metric['mean_iou']:.1%}" if len(gt_boxes) > 0 else "N/A"

        table_rows.append({
            "Tên ảnh": path.name,
            "GT": gt_str,
            "Pred": num_preds,
            "Precision": prec_str,
            "Recall": rec_str,
            "Mean IoU": iou_str,
            "Latency (ms)": f"{res['latency_ms']:.1f}"
        })

        vis_img = draw_predictions_and_gt(
            image=pil_img,
            pred_boxes=res["boxes"],
            pred_scores=res["scores"],
            gt_boxes=gt_boxes if len(gt_boxes) > 0 else None,
            show_gt=True
        )
        gallery_results.append(vis_img)

        export_name = f"{idx:04d}_{path.stem}.jpg"
        pil_img.save(yolo_img / export_name, quality=95)

        yolo_lines = []
        for box, score in zip(res["boxes"], res["scores"]):
            x1, y1, x2, y2 = map(float, box)
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) / (2.0 * w)
            cy = (y1 + y2) / (2.0 * h)
            norm_w = bw / float(w)
            norm_h = bh / float(h)
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")

            coco_dataset["annotations"].append({
                "id": ann_id_counter,
                "image_id": idx,
                "category_id": 1,
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "score": float(score),
                "iscrowd": 0
            })
            ann_id_counter += 1

        label_txt_path = yolo_lbl / f"{Path(export_name).stem}.txt"
        label_txt_path.write_text("\n".join(yolo_lines), encoding="utf-8")

        coco_dataset["images"].append({
            "id": idx,
            "file_name": export_name,
            "width": w,
            "height": h
        })

    (yolo_dir / "classes.txt").write_text("window\n", encoding="utf-8")
    coco_json_path = coco_dir / "annotations.coco.json"
    coco_json_path.write_text(json.dumps(coco_dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    yolo_zip_path = EXPORT_ROOT / f"yolo_labels_{run_id}.zip"
    with zipfile.ZipFile(yolo_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in yolo_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(yolo_dir))

    coco_zip_path = EXPORT_ROOT / f"coco_labels_{run_id}.zip"
    with zipfile.ZipFile(coco_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in coco_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(coco_dir))

    progress(1.0, desc="Hoàn tất đánh giá!")

    avg_latency = float(np.mean(times)) if times else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    hardware = f"{torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else "CPU"

    if total_gt_count > 0:
        overall_prec = total_tp / max(1, total_tp + total_fp)
        overall_rec = total_tp / max(1, total_tp + total_fn)
        overall_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0
        coco_res = evaluate_dataset_coco(records)
        val_map = f"{coco_res['mAP50_95']:.1%}"
        val_ap50 = f"{coco_res['AP50']:.1%}"
        val_ap75 = f"{coco_res['AP75']:.1%}"
        val_prec = f"{overall_prec:.1%}"
        val_rec = f"{overall_rec:.1%}"
        val_iou = f"{overall_iou:.1%}"
    else:
        val_map = "N/A"
        val_ap50 = "N/A"
        val_ap75 = "N/A"
        val_prec = "N/A"
        val_rec = "N/A"
        val_iou = "N/A"

    val_fps = f"{fps:.1f} FPS"

    df_details = pd.DataFrame(table_rows)

    status_msg = (
        f"✅ **Phân tích thành công {total_files} ảnh!**\n\n"
        f"| Thông số | Giá trị |\n|---|---|\n"
        f"| 🪟 Cửa sổ nhận diện (Pred) | **{total_pred_count:,}** |\n"
        f"| 🏷️ Ground Truth | **{total_gt_count:,}** |\n"
        f"| 💻 Phần cứng | `{hardware}` ({DEVICE.upper()}) |\n"
        f"| ⚡ Tốc độ | `{avg_latency:.1f} ms/ảnh` ≈ **{fps:.1f} FPS** |"
    )

    return (
        gallery_results,
        str(yolo_zip_path),
        str(coco_zip_path),
        df_details,
        status_msg,
        val_prec,
        val_rec,
        val_ap50,
        val_ap75,
        val_map,
        val_iou,
        val_fps
    )


def load_demo_action():
    paths = get_demo_image_paths()
    return paths


# ============================================================
# MATTE DARK / DEEP CHARCOAL THEME (#121212 & #18181B)
# Sleek architectural dark mode with vivid teal/cyan accents
# ============================================================
GRADIO_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.zinc,
    font=[gr.themes.GoogleFont("DM Sans"), gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#121212",
    block_background_fill="#18181b",
    block_border_color="#27272a",
    block_border_width="1px",
    block_label_text_color="#a1a1aa",
    block_title_text_color="#f4f4f5",
    input_background_fill="#18181b",
    input_border_color="#27272a",
    button_primary_background_fill="linear-gradient(135deg, #0d9488 0%, #14b8a6 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #0f766e 0%, #0d9488 100%)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#27272a",
    button_secondary_border_color="#3f3f46",
    button_secondary_text_color="#f4f4f5",
    shadow_drop="none",
    shadow_spread="0px",
    block_shadow="none",
)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

* {
    font-family: 'DM Sans', system-ui, -apple-system, sans-serif !important;
    box-sizing: border-box;
}

/* ── Page Background ── */
gradio-app, body, .gradio-container {
    background-color: #121212 !important;
    background: #121212 !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    color: #f4f4f5 !important;
}

/* ── Hero Header ── */
#hero-header {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 16px;
    padding: 32px 38px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

#hero-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, transparent 79px, rgba(255,255,255,0.03) 79px, rgba(255,255,255,0.03) 80px, transparent 80px),
        linear-gradient(0deg, transparent 79px, rgba(255,255,255,0.03) 79px, rgba(255,255,255,0.03) 80px, transparent 80px);
    background-size: 80px 80px;
    pointer-events: none;
}

#hero-header::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 260px; height: 100%;
    background: radial-gradient(circle at right, rgba(20, 184, 166, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

#hero-header h1 {
    color: #fafafa !important;
    font-size: 24px !important;
    font-weight: 800 !important;
    margin: 0 0 8px !important;
    letter-spacing: -0.025em;
    position: relative;
    z-index: 1;
}

#hero-header .subtitle {
    color: #a1a1aa !important;
    font-size: 14px !important;
    margin: 0 0 4px !important;
    line-height: 1.6;
    position: relative;
    z-index: 1;
}

#hero-header .subtitle strong {
    color: #2dd4bf !important;
}

#hero-header .subtitle code {
    background: #27272a;
    color: #e4e4e7;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12.5px;
    border: 1px solid #3f3f46;
}

#hero-header .tag-row {
    margin-top: 14px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    position: relative;
    z-index: 1;
}

#hero-header .tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #27272a;
    border: 1px solid #3f3f46;
    color: #e4e4e7;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 8px;
}

#hero-header .tag .icon { color: #2dd4bf; }

/* ── Section Titles ── */
.section-label {
    color: #f4f4f5 !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    margin: 0 0 14px !important;
    padding-bottom: 10px;
    border-bottom: 2px solid #27272a;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Panels ── */
#control-panel, #results-panel {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 14px !important;
    padding: 24px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
}

/* ── Demo Button ── */
#demo-btn button {
    background: #27272a !important;
    border: 1px solid #3f3f46 !important;
    color: #f4f4f5 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
}

#demo-btn button:hover {
    background: #3f3f46 !important;
    border-color: #52525b !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}

/* ── Run Button ── */
#run-btn button {
    background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 12px !important;
    padding: 14px 20px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    letter-spacing: 0.01em;
    box-shadow: 0 4px 16px rgba(13, 148, 136, 0.35) !important;
}

#run-btn button:hover {
    background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 22px rgba(13, 148, 136, 0.5) !important;
}

/* ── File Uploader ── */
.file-upload-container {
    border: 2px dashed #3f3f46 !important;
    background: #18181b !important;
    border-radius: 12px !important;
    transition: border-color 0.15s ease !important;
}
.file-upload-container:hover {
    border-color: #14b8a6 !important;
}

/* ── Status Box ── */
#status-box {
    background: #1e1e24 !important;
    border: 1px solid #27272a !important;
    border-left: 3px solid #14b8a6 !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
    color: #d4d4d8 !important;
    font-size: 13.5px !important;
    line-height: 1.65 !important;
    margin-top: 14px !important;
}
#status-box strong { color: #f4f4f5 !important; }
#status-box table { width: 100%; margin-top: 8px; border-collapse: collapse; }
#status-box th, #status-box td {
    padding: 5px 10px;
    border-bottom: 1px solid #27272a;
    color: #d4d4d8 !important;
    font-size: 13px;
}

/* ── Metric Cards ── */
.metric-card-wrap .label-wrap { display: none !important; }
.metric-card-wrap textarea, .metric-card-wrap input {
    background: transparent !important;
    border: none !important;
    color: #2dd4bf !important;
    font-size: 22px !important;
    font-weight: 800 !important;
    text-align: center !important;
    padding: 0 !important;
    box-shadow: none !important;
    cursor: default !important;
    pointer-events: none;
    font-family: 'JetBrains Mono', monospace !important;
    white-space: nowrap !important;
}

.metric-card {
    background: #1e1e24;
    border: 1px solid #27272a;
    border-radius: 12px;
    padding: 18px 10px 14px;
    text-align: center;
    transition: all 0.15s ease;
    position: relative;
}

.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 12px; right: 12px;
    height: 2px;
    background: #14b8a6;
    border-radius: 2px;
    opacity: 0;
    transition: opacity 0.15s ease;
}

.metric-card:hover {
    border-color: #3f3f46;
    background: #23232a;
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
}

.metric-card:hover::after {
    opacity: 1;
}

.metric-title {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #a1a1aa !important;
    margin-bottom: 6px;
}

/* ── Gallery ── */
.gallery-container {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 10px;
    overflow: hidden;
}

/* ── Dataframe Table ── */
.dataframe-container table {
    background: #18181b !important;
    color: #f4f4f5 !important;
    font-size: 13px !important;
    border: 1px solid #27272a !important;
    border-radius: 8px !important;
    overflow: hidden;
}
.dataframe-container th {
    background: #27272a !important;
    color: #e4e4e7 !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    border-bottom: 2px solid #3f3f46 !important;
}
.dataframe-container tr:hover td {
    background: #23232a !important;
}
.dataframe-container td {
    border-color: #27272a !important;
    color: #d4d4d8 !important;
}

/* ── Download Area ── */
.download-row .file-preview {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 10px !important;
}

/* ── Legend color swatches ── */
.legend-pred { color: #f87171; font-weight: 700; }
.legend-gt { color: #22d3ee; font-weight: 700; }
"""

with gr.Blocks(
    title="Zero-Shot Window Detection — Grounding DINO",
) as demo:

    # ── Hero Banner ──
    gr.HTML("""
    <div id="hero-header">
        <h1>Zero-Shot Facade Window Detection</h1>
        <p class="subtitle">Nhận diện cửa sổ mặt đứng kiến trúc với <strong>Grounding DINO</strong> · Model <code>grounding-dino-base</code> · Chạy hoàn toàn Offline</p>
        <p class="subtitle">Đo lường đầy đủ <strong>Precision · Recall · AP50 · AP75 · mAP50:95 · IoU · FPS</strong> · Xuất nhãn <strong>YOLO / COCO</strong></p>
        <div class="tag-row">
            <span class="tag"><span class="icon">⚡</span> GPU / CPU</span>
            <span class="tag"><span class="icon">🔒</span> 100% Offline</span>
            <span class="tag"><span class="icon">🏷</span> Auto Label Export</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT: Control Panel ──
        with gr.Column(scale=1, min_width=320):
            gr.HTML('<div id="control-panel">')
            gr.HTML('<p class="section-label">Dữ liệu & Điều khiển</p>')

            with gr.Group(elem_id="demo-btn"):
                demo_btn = gr.Button("📁 Nạp 5 Ảnh Mẫu Demo", size="md")

            file_input = gr.File(
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                label="Tải lên ảnh (chọn một hoặc nhiều ảnh)",
                elem_classes=["file-upload-container"]
            )

            with gr.Group(elem_id="run-btn"):
                run_btn = gr.Button("🚀 BẮT ĐẦU NHẬN DIỆN & TÍNH METRICS", variant="primary", size="lg")

            status_box = gr.Markdown(
                value="*Chưa có dữ liệu phân tích. Nhấn nút phía trên để bắt đầu.*",
                elem_id="status-box"
            )

            gr.HTML('<p class="section-label" style="margin-top:22px">Xuất file nhãn</p>')
            with gr.Row(elem_classes=["download-row"]):
                yolo_download = gr.File(label="YOLO (.zip)")
                coco_download = gr.File(label="COCO (.zip)")

            gr.HTML('</div>')

        # ── RIGHT: Results Panel ──
        with gr.Column(scale=3):
            gr.HTML('<div id="results-panel">')
            gr.HTML('<p class="section-label">Metrics Dashboard</p>')

            # Row 1: Precision · Recall · IoU · FPS
            with gr.Row():
                with gr.Column(min_width=130):
                    gr.HTML('<div class="metric-card"><div class="metric-title">Precision @0.50</div>')
                    card_prec = gr.Textbox(value="-", interactive=False, max_lines=1,
                                           show_label=False, elem_classes=["metric-card-wrap"])
                    gr.HTML('</div>')
                with gr.Column(min_width=130):
                    gr.HTML('<div class="metric-card"><div class="metric-title">Recall @0.50</div>')
                    card_rec = gr.Textbox(value="-", interactive=False, max_lines=1,
                                          show_label=False, elem_classes=["metric-card-wrap"])
                    gr.HTML('</div>')
                with gr.Column(min_width=130):
                    gr.HTML('<div class="metric-card"><div class="metric-title">Mean IoU</div>')
                    card_iou = gr.Textbox(value="-", interactive=False, max_lines=1,
                                          show_label=False, elem_classes=["metric-card-wrap"])
                    gr.HTML('</div>')
                with gr.Column(min_width=130):
                    gr.HTML('<div class="metric-card"><div class="metric-title">Throughput</div>')
                    card_fps = gr.Textbox(value="-", interactive=False, max_lines=1,
                                          show_label=False, elem_classes=["metric-card-wrap"])
                    gr.HTML('</div>')

            # Row 2: AP50 · AP75 · mAP
            with gr.Row():
                with gr.Column(min_width=130):
                    gr.HTML('<div class="metric-card"><div class="metric-title">AP50 (COCO)</div>')
                    card_ap50 = gr.Textbox(value="-", interactive=False, max_lines=1,
                                           show_label=False, elem_classes=["metric-card-wrap"])
                    gr.HTML('</div>')
                with gr.Column(min_width=130):
                    gr.HTML('<div class="metric-card"><div class="metric-title">AP75 (COCO)</div>')
                    card_ap75 = gr.Textbox(value="-", interactive=False, max_lines=1,
                                           show_label=False, elem_classes=["metric-card-wrap"])
                    gr.HTML('</div>')
                with gr.Column(min_width=130):
                    gr.HTML('<div class="metric-card"><div class="metric-title">mAP @ [50:95]</div>')
                    card_map = gr.Textbox(value="-", interactive=False, max_lines=1,
                                          show_label=False, elem_classes=["metric-card-wrap"])
                    gr.HTML('</div>')
                with gr.Column(min_width=130):
                    gr.HTML('<div style="height:10px"></div>')

            gr.HTML('<p class="section-label" style="margin-top:22px">Kết quả trực quan &nbsp;<span style="font-size:12px;font-weight:600;letter-spacing:0;text-transform:none"><span class="legend-pred">■ Đỏ = Prediction</span> &nbsp;·&nbsp; <span class="legend-gt">■ Xanh = Ground Truth</span></span></p>')
            gallery_view = gr.Gallery(
                label="",
                columns=2,
                height=480,
                object_fit="contain",
                preview=True,
                elem_classes=["gallery-container"]
            )

            gr.HTML('<p class="section-label" style="margin-top:22px">Chi tiết từng ảnh</p>')
            table_view = gr.Dataframe(
                headers=["Tên ảnh", "GT", "Pred", "Precision", "Recall", "Mean IoU", "Latency (ms)"],
                interactive=False,
                wrap=True,
                elem_classes=["dataframe-container"]
            )
            gr.HTML('</div>')

    # ── Event bindings ──
    demo_btn.click(fn=load_demo_action, outputs=[file_input])

    run_btn.click(
        fn=run_pipeline,
        inputs=[file_input],
        outputs=[
            gallery_view, yolo_download, coco_download, table_view,
            status_box, card_prec, card_rec, card_ap50, card_ap75, card_map, card_iou, card_fps
        ]
    )


if __name__ == "__main__":
    share_flag = os.getenv("GRADIO_SHARE", "False").lower() in ("true", "1")
    inbrowser_flag = os.getenv("GRADIO_INBROWSER", "False").lower() in ("true", "1")
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share_flag,
        inbrowser=inbrowser_flag,
        theme=GRADIO_THEME,
        css=custom_css,
    )
