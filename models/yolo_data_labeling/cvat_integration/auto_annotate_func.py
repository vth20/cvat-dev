"""
CVAT auto-annotation function cho model YOLO đã fine-tune (ultralytics .pt).

Dùng:
    cvat-cli --server-host http://localhost:8080 --auth <user> \
        task auto-annotate <TASK_ID> \
        --function-file cvat_integration/auto_annotate_func.py \
        -p model_path=str:Yolo/model/yolov11s_best_v2.pt \
        --conf-threshold 0.25 --allow-unmatched-labels

Tên label trong task CVAT phải trùng tên class của model (ở đây: "window").
"""

from __future__ import annotations

import PIL.Image
import torch
from ultralytics import YOLO

import cvat_sdk.auto_annotation as cvataa
import cvat_sdk.models as models


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _YoloFunction:
    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        imgsz: int = 640,
        iou: float = 0.45,
        max_det: int = 1000,
        default_conf: float = 0.25,
    ) -> None:
        self._model = YOLO(model_path)
        self._device = device or _default_device()
        self._imgsz = imgsz
        self._iou = iou
        self._max_det = max_det
        self._default_conf = default_conf

    @property
    def spec(self) -> cvataa.DetectionFunctionSpec:
        return cvataa.DetectionFunctionSpec(
            labels=[
                cvataa.label_spec(name, int(class_id), type="rectangle")
                for class_id, name in self._model.names.items()
            ]
        )

    def detect(
        self, context: cvataa.DetectionFunctionContext, image: PIL.Image.Image
    ) -> list[models.LabeledShapeRequest]:
        conf = context.conf_threshold if context.conf_threshold is not None else self._default_conf

        result = self._model.predict(
            image.convert("RGB"),
            conf=conf,
            iou=self._iou,
            imgsz=self._imgsz,
            max_det=self._max_det,
            device=self._device,
            verbose=False,
        )[0]

        return [
            cvataa.rectangle(int(cls), [float(v) for v in xyxy])
            for xyxy, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist())
        ]


def create(**kwargs) -> cvataa.DetectionFunction:
    return _YoloFunction(**kwargs)
