"""Nuclio handler cho CVAT: Grounding DINO pretrained (zero-shot window detection).

Tái sử dụng load_detector / predict_single_image trong inference.py của repo
(deploy_nuclio.sh copy file này vào /opt/nuclio lúc build).
"""

import base64
import io
import json
import os
from pathlib import Path

import torch
from PIL import Image

from inference import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_NMS_IOU,
    DEFAULT_TEXT_THRESHOLD,
    load_detector,
    predict_single_image,
)

HF_MODEL_ID = "IDEA-Research/grounding-dino-base"
LOCAL_MODEL_DIR = Path("/opt/nuclio/saved_model")

# Mỗi label vừa là prompt cho model, vừa phải khớp với metadata.annotations.spec trong function.yaml
LABELS = [name.strip() for name in os.environ.get("LABELS", "window").split(",") if name.strip()]
TEXT_THRESHOLD = float(os.environ.get("TEXT_THRESHOLD", DEFAULT_TEXT_THRESHOLD))
NMS_IOU = float(os.environ.get("NMS_IOU", DEFAULT_NMS_IOU))


def _match_label(phrase):
    # Grounding DINO trả về cụm từ khớp với prompt (vd: "window"), map về label gần nhất
    phrase = str(phrase).strip().lower()
    for name in LABELS:
        if name.lower() == phrase:
            return name
    for name in LABELS:
        if name.lower() in phrase or phrase in name.lower():
            return name
    return LABELS[0] if len(LABELS) == 1 else None


def init_context(context):
    context.logger.info("Init context...  0%")

    model_path = LOCAL_MODEL_DIR if (LOCAL_MODEL_DIR / "config.json").exists() else HF_MODEL_ID
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model, dtype, device = load_detector(model_path, device=device)

    context.user_data.processor = processor
    context.user_data.model = model
    context.user_data.dtype = dtype
    context.user_data.device = device
    context.user_data.prompt = " ".join(f"{name}." for name in LABELS)

    context.logger.info(f"Loaded {model_path} on {device}, prompt: {context.user_data.prompt!r}")
    context.logger.info("Init context...100%")


def handler(context, event):
    data = event.body
    image = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
    threshold = float(data.get("threshold") or DEFAULT_BOX_THRESHOLD)

    ud = context.user_data
    pred = predict_single_image(
        image,
        ud.processor,
        ud.model,
        prompt=ud.prompt,
        box_threshold=threshold,
        text_threshold=TEXT_THRESHOLD,
        nms_iou=NMS_IOU,
        dtype=ud.dtype,
        device=ud.device,
    )

    results = []
    for box, score, phrase in zip(pred["boxes"], pred["scores"], pred["labels"]):
        label = _match_label(phrase)
        if label is None:
            continue
        results.append(
            {
                "confidence": str(float(score)),
                "label": label,
                "points": [float(v) for v in box],
                "type": "rectangle",
            }
        )

    return context.Response(
        body=json.dumps(results), headers={}, content_type="application/json", status_code=200
    )
