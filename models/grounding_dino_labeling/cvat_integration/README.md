# Tích hợp CVAT — Grounding DINO Window (pretrained)

Hai cách đưa Grounding DINO (`IDEA-Research/grounding-dino-base`) vào CVAT Community:

| Cách | Dùng khi | File |
| ---- | -------- | ---- |
| Nuclio function | Muốn dùng model ngay trong UI (AI Tools → Detectors, Actions → Automatic annotation) | `nuclio/`, `deploy_nuclio.sh` |
| `cvat-cli task auto-annotate` | Chạy hàng loạt từ máy local, không cần Nuclio | `auto_annotate_func.py` |

Cả hai đều dùng lại `load_detector` / `predict_single_image` trong `inference.py`
(giữ nguyên lọc diện tích + NMS).

## 1. Nuclio function (trên server CVAT)

Yêu cầu:
- CVAT chạy kèm serverless component (có container `nuclio`): ở thư mục gốc repo chạy `./scripts/cvat.sh`.
  Muốn deploy tất cả model cùng lúc: `./scripts/deploy_models.sh`.
- Server có internet lúc build (tải torch, transformers, nuctl và ~900MB trọng số từ Hugging Face).
- GPU: đã cài NVIDIA driver + NVIDIA Container Toolkit. Nên dùng GPU; trên CPU mất vài giây/ảnh.
- RAM trống ≥ 3GB cho function (đo được ~2.5GB lúc suy luận trên CPU). Thiếu RAM thì log sẽ báo
  `Unexpected termination of child process ... signal: killed`.

Deploy:

```bash
cd models/grounding_dino_labeling
./cvat_integration/deploy_nuclio.sh          # tự chọn GPU/CPU
./cvat_integration/deploy_nuclio.sh --cpu
```

Script sẽ:
1. Tự tải `nuctl` đúng version với container `nuclio` (lưu ở `cvat_integration/.bin/`).
2. Gom `main.py` + `inference.py` (và `saved_model/` nếu có) vào thư mục build tạm.
3. Build image `cvat.custom.grounding-dino-window` (trọng số được tải sẵn vào image,
   function chạy offline) và deploy function `custom-grounding-dino-window` vào network `cvat_cvat`
   (đổi bằng `CVAT_NETWORK=<tên> ./cvat_integration/deploy_nuclio.sh` nếu network khác).

Sau khi deploy, vào CVAT → **Models** sẽ thấy **Grounding DINO Window (pretrained)**.
Label trong task phải tên `window` (hoặc map lại khi chạy automatic annotation).

Thêm label/prompt khác (vd `window,door`): sửa **cả hai** chỗ trong `nuclio/function*.yaml`
- `spec.env` → `LABELS: window,door`
- `metadata.annotations.spec` → thêm `{ "id": 1, "name": "door", "type": "rectangle" }`

rồi chạy lại script. Các tham số khác trong `spec.env`: `TEXT_THRESHOLD`, `NMS_IOU`.
Box threshold lấy từ ngưỡng CVAT gửi sang (mặc định 0.22).

Function được cố định host port **32102** (`triggers.myHttpTrigger.attributes.port`) để CVAT
vẫn gọi đúng khi container restart. Nếu port này đã bị dùng trên server, đổi sang port khác
trong cả `function.yaml` và `function-gpu.yaml`.

Xem log khi lỗi:

```bash
docker logs nuclio-nuclio-custom-grounding-dino-window
```

## 2. cvat-cli auto-annotate (local)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install "cvat-cli==<phiên bản CVAT server>"   # vd: cvat-cli==2.76.*
pip install torch torchvision "transformers>=4.40,<=4.44.2" timm
export CVAT_ACCESS_TOKEN=<token>   # CVAT → Settings → Access tokens

cvat-cli --server-host http://<cvat-host>:8080 task auto-annotate <TASK_ID> \
    --function-file cvat_integration/auto_annotate_func.py \
    -p labels=str:window \
    --conf-threshold 0.22 --allow-unmatched-labels
```

Thêm `--clear-existing` để xoá annotation cũ của task trước khi ghi.
