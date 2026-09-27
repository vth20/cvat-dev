# CVAT + AI auto-annotation (window detection)

CVAT Community (hiện tại v2.76.0, xem submodule `cvat/`) kèm 2 model tự động gán nhãn cửa sổ, chạy dưới dạng Nuclio function:

| Model | Tên trong CVAT | Port |
| ----- | -------------- | ---- |
| YOLO fine-tune (`models/yolo_data_labeling`) | YOLO Window (fine-tuned) | 32101 |
| Grounding DINO pretrained (`models/grounding_dino_labeling`) | Grounding DINO Window (pretrained) | 32102 |

## Cấu trúc

```text
.
├── cvat/                     # submodule: github.com/cvat-ai/cvat, ghim theo commit (không sửa)
├── models/
│   ├── yolo_data_labeling/       # notebook train, model .pt, cvat_integration/
│   └── grounding_dino_labeling/  # inference, gradio app, cvat_integration/
├── scripts/
│   ├── cvat.sh               # docker compose CVAT + serverless
│   ├── deploy_models.sh      # build & deploy tất cả Nuclio function
│   └── upgrade_cvat.sh       # nâng cấp CVAT trên server (backup + migrate + deploy lại model)
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

## Cập nhật model / code AI

```bash
git pull
./scripts/deploy_models.sh    # build lại model (có cache, nhanh hơn lần đầu)
```

Đổi model YOLO: `models/yolo_data_labeling/cvat_integration/deploy_nuclio.sh Yolo/model/<file>.pt`.

## Nâng cấp CVAT

Phiên bản CVAT = commit của submodule `cvat/`. Image Docker lấy theo phiên bản ghi trong
`cvat/docker-compose.yml`, nên không đặt `CVAT_VERSION` trong `.env` (trừ khi cố ý ghim).

**1. Trên máy dev: nâng submodule rồi commit**

```bash
git -C cvat fetch --depth 1 origin tag v2.77.0     # thay bằng tag muốn lên
git -C cvat checkout v2.77.0
git add cvat
git commit -m "Upgrade CVAT to v2.77.0"
git push
```

Trước khi push nên:
- Đọc `cvat/CHANGELOG.md` và mục phiên bản tương ứng trong
  [upgrade guide](https://docs.cvat.ai/docs/administration/community/advanced/upgrade_guide/).
- Xem `cvat/components/serverless/docker-compose.serverless.yml` có đổi version Nuclio không
  (script deploy tự tải `nuctl` khớp version, chỉ cần deploy lại model).
- Nếu dùng `cvat-cli task auto-annotate`: cài `cvat-cli` cùng phiên bản với server.

**2. Trên server**

```bash
git pull
./scripts/upgrade_cvat.sh
```

Script sẽ: backup DB + data + events vào `backups/<thời gian>/` → cập nhật submodule →
pull image mới → khởi động (CVAT tự migrate DB, **không tắt giữa chừng**) → chờ CVAT sẵn sàng →
deploy lại 2 model.

Quay lại bản cũ khi lỗi: `git checkout <commit cũ> && git submodule update --init --depth 1`,
khôi phục backup theo [backup guide](https://docs.cvat.ai/docs/administration/community/advanced/backup_guide/)
(DB đã migrate lên schema mới không dùng được với bản cũ), rồi `./scripts/cvat.sh up -d`.

## Xử lý lỗi

| Hiện tượng | Nguyên nhân / cách xử lý |
| ---------- | ------------------------ |
| Menu **Models** không hiện | Trang được mở khi server chưa sẵn sàng → tải lại trang. Kiểm tra `docker ps` có container `nuclio`. |
| Automatic annotation báo `RemoteDisconnected` / `Connection aborted` | Function bị kill, thường do thiếu RAM. Xem `docker logs nuclio-nuclio-custom-grounding-dino-window` có `signal: killed`. |
| Automatic annotation báo `http://host.docker.internal:None` | Function không ở trạng thái ready (build lỗi, container không khởi động, thiếu NVIDIA Container Toolkit khi dùng GPU). Chạy lại `./scripts/deploy_models.sh` và đọc lỗi; xem `docker logs --tail 100 nuclio-nuclio-<tên function>`. |
| Deploy báo không có network `cvat_cvat` | Đặt `CVAT_NETWORK` trong `.env` theo `docker network ls`. |
| Port 32101/32102 bị trùng | Đổi `triggers.myHttpTrigger.attributes.port` trong `function.yaml` và `function-gpu.yaml` của model đó. |

Log:

```bash
./scripts/cvat.sh logs -f cvat_server cvat_worker_annotation
docker logs -f nuclio-nuclio-custom-yolo-window
docker logs -f nuclio-nuclio-custom-grounding-dino-window
```
