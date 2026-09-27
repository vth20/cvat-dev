#!/usr/bin/env bash
# Build & deploy Nuclio function "YOLO Window" lên CVAT. Chạy trên máy đang chạy CVAT.
#
#   ./cvat_integration/deploy_nuclio.sh                 # tự chọn GPU nếu có GPU + NVIDIA Container Toolkit, không thì CPU
#   ./cvat_integration/deploy_nuclio.sh --cpu
#   ./cvat_integration/deploy_nuclio.sh --gpu Yolo/model/yolov11m_best.pt
#
# Biến môi trường tuỳ chọn:
#   CVAT_NETWORK   docker network của CVAT (mặc định: cvat_cvat)
#   NUCTL          đường dẫn nuctl có sẵn (mặc định: tự tải đúng version của container nuclio)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DEFAULT_MODEL="$REPO_DIR/Yolo/model/yolov11s_best_v2.pt"
CVAT_NETWORK="${CVAT_NETWORK:-cvat_cvat}"

MODE=auto
MODEL=""
for arg in "$@"; do
    case "$arg" in
        --cpu) MODE=cpu ;;
        --gpu) MODE=gpu ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) MODEL="$arg" ;;
    esac
done
MODEL="$(realpath "${MODEL:-$DEFAULT_MODEL}")"

die() { echo "ERROR: $*" >&2; exit 1; }

[ -f "$MODEL" ] || die "Không tìm thấy model: $MODEL"

has_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }
# Docker phải được cấu hình NVIDIA Container Toolkit thì container mới dùng được GPU
docker_has_gpu() { docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; }

if [ "$MODE" = auto ]; then
    if has_gpu && docker_has_gpu; then
        MODE=gpu
    else
        has_gpu && echo "WARN: Có GPU nhưng Docker chưa cấu hình NVIDIA Container Toolkit -> dùng CPU" >&2
        MODE=cpu
    fi
elif [ "$MODE" = gpu ]; then
    docker_has_gpu || die "Docker chưa cấu hình NVIDIA Container Toolkit (không có runtime 'nvidia' trong 'docker info')"
fi
if [ "$MODE" = gpu ]; then
    FUNC_YAML="$SCRIPT_DIR/nuclio/function-gpu.yaml"
else
    FUNC_YAML="$SCRIPT_DIR/nuclio/function.yaml"
fi

# --- Kiểm tra CVAT serverless ---
command -v docker >/dev/null 2>&1 || die "Chưa cài docker"
[ "$(docker inspect -f '{{.State.Running}}' nuclio 2>/dev/null)" = true ] || die "Container 'nuclio' chưa chạy. Ở thư mục gốc repo, chạy: ./scripts/cvat.sh"
docker network inspect "$CVAT_NETWORK" >/dev/null 2>&1 || die "Không có docker network '$CVAT_NETWORK'. \
Xem 'docker network ls' rồi đặt CVAT_NETWORK=<tên network>"

# --- nuctl (phải cùng version với nuclio dashboard) ---
if [ -z "${NUCTL:-}" ]; then
    NUCLIO_VERSION="$(docker inspect nuclio --format '{{.Config.Image}}' | sed -E 's/.*:([0-9.]+).*/\1/')"
    case "$(uname -m)" in
        x86_64|amd64) NUCTL_ARCH=amd64 ;;
        aarch64|arm64) NUCTL_ARCH=arm64 ;;
        *) die "Không hỗ trợ kiến trúc $(uname -m)" ;;
    esac
    NUCTL_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
    NUCTL="$SCRIPT_DIR/.bin/nuctl-$NUCLIO_VERSION"
    if [ ! -x "$NUCTL" ]; then
        echo "Tải nuctl $NUCLIO_VERSION ($NUCTL_OS-$NUCTL_ARCH)..."
        mkdir -p "$SCRIPT_DIR/.bin"
        curl -fL -o "$NUCTL" \
            "https://github.com/nuclio/nuclio/releases/download/$NUCLIO_VERSION/nuctl-$NUCLIO_VERSION-$NUCTL_OS-$NUCTL_ARCH"
        chmod +x "$NUCTL"
    fi
fi

# --- Gom source cần thiết vào thư mục build tạm ---
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

cp "$SCRIPT_DIR/nuclio/main.py" "$BUILD_DIR/"
cp "$FUNC_YAML" "$BUILD_DIR/function.yaml"
cp "$MODEL" "$BUILD_DIR/model.pt"

echo "Deploy YOLO Window ($MODE) với model $(basename "$MODEL")..."
"$NUCTL" create project cvat --platform local >/dev/null 2>&1 || true
"$NUCTL" deploy --project-name cvat \
    --path "$BUILD_DIR" \
    --file "$BUILD_DIR/function.yaml" \
    --platform local \
    --platform-config "{\"attributes\": {\"network\": \"$CVAT_NETWORK\"}}"

"$NUCTL" get function --platform local

# --- Kiểm tra function thực sự chạy được (CVAT cần function ở trạng thái ready và có port) ---
FUNC_NAME=custom-yolo-window
CONTAINER="nuclio-nuclio-$FUNC_NAME"
# Ảnh PNG trắng 64x64
SMOKE_IMAGE="iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PMQ0AAAwDoPo33UrYvQQckD4XAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAYHLAMpT0sIcNbcEAAAAAElFTkSuQmCC"

"$NUCTL" get function "$FUNC_NAME" --platform local | grep -w ready >/dev/null \
    || die "Function $FUNC_NAME không ở trạng thái ready. Xem: docker logs --tail 100 $CONTAINER"
PORT="$(docker port "$CONTAINER" 8080/tcp 2>/dev/null | head -1 | sed 's/.*://')"
[ -n "$PORT" ] || die "Container $CONTAINER không chạy. Xem: docker logs --tail 100 $CONTAINER"

# Nuclio báo ready trước khi model load xong -> thử lại tối đa ~5 phút
echo "Gọi thử function ở port $PORT..."
for _ in $(seq 1 30); do
    if curl -sf -m 300 -X POST "http://localhost:$PORT" -H 'Content-Type: application/json' \
        -d "{\"image\": \"$SMOKE_IMAGE\"}" >/dev/null 2>&1; then
        echo "OK: $FUNC_NAME hoạt động (port $PORT)."
        exit 0
    fi
    sleep 10
done
die "Gọi thử function lỗi. Xem: docker logs --tail 100 $CONTAINER"
