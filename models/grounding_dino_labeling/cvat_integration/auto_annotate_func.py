"""
CVAT auto-annotation function cho Grounding DINO pretrained (zero-shot, theo text prompt).
Tái sử dụng load_detector / predict_single_image trong inference.py ở thư mục gốc repo.

Dùng:
    cvat-cli --server-host http://localhost:8080 --auth <user> \
        task auto-annotate <TASK_ID> \
        --function-file cvat_integration/auto_annotate_func.py \
        -p labels=str:window \
        --conf-threshold 0.22 --allow-unmatched-labels

`labels` là danh sách tên label, cách nhau bởi dấu phẩy (vd: "window,door").
Mỗi tên vừa là prompt cho model, vừa phải trùng tên label trong task CVAT.
"""

from __future__ import annotations

import sys
from pathlib import Path

import PIL.Image

_REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_DIR))

from inference import (  # noqa: E402
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_NMS_IOU,
    DEFAULT_TEXT_THRESHOLD,
    load_detector,
    predict_single_image,
)

import cvat_sdk.auto_annotation as cvataa  # noqa: E402
import cvat_sdk.models as models  # noqa: E402

_HF_MODEL_ID = "IDEA-Research/grounding-dino-base"


def _default_model_path() -> str:
    local_saved = _REPO_DIR / "saved_model"
    if (local_saved / "config.json").exists():
        return str(local_saved)
    return _HF_MODEL_ID


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _GroundingDinoFunction:
    def __init__(
        self,
        labels: str = "window",
        model_path: str | None = None,
        device: str | None = None,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        nms_iou: float = DEFAULT_NMS_IOU,
    ) -> None:
        self._labels = [name.strip() for name in labels.split(",") if name.strip()]
        self._prompt = " ".join(f"{name}." for name in self._labels)
        self._text_threshold = text_threshold
        self._nms_iou = nms_iou

        self._processor, self._model, self._dtype, self._device = load_detector(
            model_path or _default_model_path(), device=device or _default_device()
        )

    @property
    def spec(self) -> cvataa.DetectionFunctionSpec:
        return cvataa.DetectionFunctionSpec(
            labels=[
                cvataa.label_spec(name, i, type="rectangle") for i, name in enumerate(self._labels)
            ]
        )

    def _label_id(self, phrase: str) -> int | None:
        # Grounding DINO trả về cụm từ khớp với prompt (vd: "window"), map về label gần nhất
        phrase = phrase.strip().lower()
        for i, name in enumerate(self._labels):
            if name.lower() == phrase:
                return i
        for i, name in enumerate(self._labels):
            if name.lower() in phrase or phrase in name.lower():
                return i
        return 0 if len(self._labels) == 1 else None

    def detect(
        self, context: cvataa.DetectionFunctionContext, image: PIL.Image.Image
    ) -> list[models.LabeledShapeRequest]:
        box_threshold = (
            context.conf_threshold
            if context.conf_threshold is not None
            else DEFAULT_BOX_THRESHOLD
        )

        pred = predict_single_image(
            image,
            self._processor,
            self._model,
            prompt=self._prompt,
            box_threshold=box_threshold,
            text_threshold=self._text_threshold,
            nms_iou=self._nms_iou,
            dtype=self._dtype,
            device=self._device,
        )

        shapes = []
        for box, label in zip(pred["boxes"], pred["labels"]):
            label_id = self._label_id(str(label))
            if label_id is None:
                continue
            shapes.append(cvataa.rectangle(label_id, [float(v) for v in box]))
        return shapes


def create(**kwargs) -> cvataa.DetectionFunction:
    return _GroundingDinoFunction(**kwargs)
