# CVAT + AI auto-annotation (window detection)

CVAT Community v2.76.0 kèm 2 model tự động gán nhãn cửa sổ, chạy dưới dạng Nuclio function:

| Model | Tên trong CVAT | Port |
| ----- | -------------- | ---- |
| YOLO fine-tune (`models/yolo_data_labeling`) | YOLO Window (fine-tuned) | 32101 |
| Grounding DINO pretrained (`models/grounding_dino_labeling`) | Grounding DINO Window (pretrained) | 32102 |

## Cấu trúc

```text
.
├── cvat/                     # submodule: github.com/cvat-ai/cvat @ v2.76.0 (không sửa)
├── models/
│   ├── yolo_data_labeling/       # notebook train, model .pt, cvat_integration/
│   └── grounding_dino_labeling/  # inference, gradio app, cvat_integration/
├── scripts/
│   ├── cvat.sh               # docker compose CVAT + serverless
│   └── deploy_models.sh      # build & deploy tất cả Nuclio function
└── .env.example
```

Mỗi `models/*/cvat_integration/` gồm Nuclio function (`nuclio/`), script deploy riêng
và file cho `cvat-cli task auto-annotate`. Xem README trong từng thư mục.

## Yêu cầu server

- Linux x86_64, Docker Engine + Docker Compose v2, user có quyền chạy `docker`.
- Internet lúc cài (pull image CVAT, build image model, tải trọng số Grounding DINO ~900MB).
- RAM: CVAT cần ~6GB; Grounding DINO cần thêm ~3GB (CPU). YOLO nhẹ (~0.6GB).
- GPU (tuỳ chọn): NVIDIA driver + NVIDIA Container Toolkit. Script tự dùng GPU nếu `nvidia-smi` chạy được.
- Mở port 8080 (CVAT UI) cho người dùng. Port 32101/32102 chỉ cần nội bộ.

## Cài đặt

```bash
git clone --recurse-submodules <repo-url> cvat-ai && cd cvat-ai
cp .env.example .env          # sửa CVAT_HOST thành IP/domain của server
./scripts/cvat.sh             # khởi động CVAT + Nuclio
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
./scripts/deploy_models.sh    # build & deploy 2 model (lần đầu mất 10-20 phút)
```

Mở `http://<CVAT_HOST>:8080`, đăng nhập, vào **Models** sẽ thấy 2 model.
Task cần có label `window`. Dùng: mở task → **Actions → Automatic annotation**, hoặc trong job
chọn **AI Tools → Detectors**.

Nếu clone thiếu `--recurse-submodules`, `./scripts/cvat.sh` sẽ tự tải submodule.

## Cập nhật

```bash
git pull
git submodule update --init --depth 1
./scripts/deploy_models.sh    # build lại model (có cache, nhanh hơn lần đầu)
```

Đổi model YOLO: `models/yolo_data_labeling/cvat_integration/deploy_nuclio.sh Yolo/model/<file>.pt`.

## Xử lý lỗi

| Hiện tượng | Nguyên nhân / cách xử lý |
| ---------- | ------------------------ |
| Menu **Models** không hiện | Trang được mở khi server chưa sẵn sàng → tải lại trang. Kiểm tra `docker ps` có container `nuclio`. |
| Automatic annotation báo `RemoteDisconnected` / `Connection aborted` | Function bị kill, thường do thiếu RAM. Xem `docker logs nuclio-nuclio-custom-grounding-dino-window` có `signal: killed`. |
| Deploy báo không có network `cvat_cvat` | Đặt `CVAT_NETWORK` trong `.env` theo `docker network ls`. |
| Port 32101/32102 bị trùng | Đổi `triggers.myHttpTrigger.attributes.port` trong `function.yaml` và `function-gpu.yaml` của model đó. |

Log:

```bash
./scripts/cvat.sh logs -f cvat_server cvat_worker_annotation
docker logs -f nuclio-nuclio-custom-yolo-window
docker logs -f nuclio-nuclio-custom-grounding-dino-window
```
