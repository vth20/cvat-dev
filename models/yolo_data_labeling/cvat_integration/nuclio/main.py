"""Nuclio handler cho CVAT: YOLO fine-tune (window detection)."""

import base64
import io
import json
import os

import torch
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/nuclio/model.pt")
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.25"))
IOU = float(os.environ.get("IOU", "0.45"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
MAX_DET = int(os.environ.get("MAX_DET", "1000"))


def init_context(context):
    context.logger.info("Init context...  0%")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    context.user_data.model = model
    context.user_data.device = device

    context.logger.info(f"Loaded {MODEL_PATH} on {device}, classes: {model.names}")
    context.logger.info("Init context...100%")


def handler(context, event):
    data = event.body
    image = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
    threshold = float(data.get("threshold") or DEFAULT_THRESHOLD)

    model = context.user_data.model
    result = model.predict(
        image,
        conf=threshold,
        iou=IOU,
        imgsz=IMGSZ,
        max_det=MAX_DET,
        device=context.user_data.device,
        verbose=False,
    )[0]

    boxes = result.boxes
    results = [
        {
            "confidence": str(conf),
            "label": model.names[int(cls)],
            "points": [float(v) for v in xyxy],
            "type": "rectangle",
        }
        for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
    ]

    return context.Response(
        body=json.dumps(results), headers={}, content_type="application/json", status_code=200
    )
