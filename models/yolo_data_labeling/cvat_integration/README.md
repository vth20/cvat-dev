# Tích hợp CVAT — YOLO Window (fine-tuned)

Hai cách đưa model YOLO đã train (`Yolo/model/*.pt`) vào CVAT Community:

| Cách | Dùng khi | File |
| ---- | -------- | ---- |
| Nuclio function | Muốn dùng model ngay trong UI (AI Tools → Detectors, Actions → Automatic annotation) | `nuclio/`, `deploy_nuclio.sh` |
| `cvat-cli task auto-annotate` | Chạy hàng loạt từ máy local, không cần Nuclio | `auto_annotate_func.py` |

## 1. Nuclio function (trên server CVAT)

Yêu cầu:
- CVAT chạy kèm serverless component (có container `nuclio`): ở thư mục gốc repo chạy `./scripts/cvat.sh`.
  Muốn deploy tất cả model cùng lúc: `./scripts/deploy_models.sh`.
- Server có internet lúc build (tải torch, ultralytics, nuctl).
- GPU: đã cài NVIDIA driver + NVIDIA Container Toolkit.

Deploy:

```bash
cd models/yolo_data_labeling
./cvat_integration/deploy_nuclio.sh                                  # tự chọn GPU/CPU, model mặc định yolov11s_best_v2.pt
./cvat_integration/deploy_nuclio.sh --gpu Yolo/model/yolov11m_best.pt  # chọn model khác
```

Script sẽ:
1. Tự tải `nuctl` đúng version với container `nuclio` (lưu ở `cvat_integration/.bin/`).
2. Gom `main.py` + file `.pt` đã chọn (đổi tên thành `model.pt`) vào thư mục build tạm.
3. Build image `cvat.custom.yolo-window` và deploy function `custom-yolo-window` vào network `cvat_cvat`
   (đổi bằng `CVAT_NETWORK=<tên> ./cvat_integration/deploy_nuclio.sh` nếu network khác).

Sau khi deploy, vào CVAT → **Models** sẽ thấy **YOLO Window (fine-tuned)**.
Label trong task phải tên `window` (hoặc map lại khi chạy automatic annotation).

Cập nhật model: train xong, commit file `.pt` mới, `git pull` trên server và chạy lại script.

Tham số chỉnh trong `nuclio/function*.yaml` → `spec.env`: `DEFAULT_THRESHOLD`, `IMGSZ`, `IOU`, `MAX_DET`.
Nếu model mới có thêm class, cập nhật `metadata.annotations.spec` cho khớp `model.names`.

Function được cố định host port **32101** (`triggers.myHttpTrigger.attributes.port`) để CVAT
vẫn gọi đúng khi container restart. Nếu port này đã bị dùng trên server, đổi sang port khác
trong cả `function.yaml` và `function-gpu.yaml`.

Xem log khi lỗi:

```bash
docker logs nuclio-nuclio-custom-yolo-window
```

## 2. cvat-cli auto-annotate (local)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install "cvat-cli==2.76.*" ultralytics torch torchvision
export CVAT_ACCESS_TOKEN=<token>   # CVAT → Settings → Access tokens

cvat-cli --server-host http://<cvat-host>:8080 task auto-annotate <TASK_ID> \
    --function-file cvat_integration/auto_annotate_func.py \
    -p model_path=str:Yolo/model/yolov11s_best_v2.pt \
    --conf-threshold 0.25 --allow-unmatched-labels
```

Thêm `--clear-existing` để xoá annotation cũ của task trước khi ghi.
