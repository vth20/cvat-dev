#!/usr/bin/env bash
# Build & deploy Nuclio function "Grounding DINO Window" lên CVAT. Chạy trên máy đang chạy CVAT.
#
#   ./cvat_integration/deploy_nuclio.sh                 # tự chọn GPU nếu có nvidia-smi, không thì CPU
#   ./cvat_integration/deploy_nuclio.sh --cpu
#
# Trọng số IDEA-Research/grounding-dino-base được tải vào image lúc build.
# Nếu repo có thư mục saved_model/ (chứa config.json) thì function sẽ dùng nó thay thế.
#
# Biến môi trường tuỳ chọn:
#   CVAT_NETWORK   docker network của CVAT (mặc định: cvat_cvat)
#   NUCTL          đường dẫn nuctl có sẵn (mặc định: tự tải đúng version của container nuclio)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CVAT_NETWORK="${CVAT_NETWORK:-cvat_cvat}"

MODE=auto
for arg in "$@"; do
    case "$arg" in
        --cpu) MODE=cpu ;;
        --gpu) MODE=gpu ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Tham số không hợp lệ: $arg" >&2; exit 1 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

if [ "$MODE" = auto ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        MODE=gpu
    else
        MODE=cpu
    fi
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
cp "$REPO_DIR/inference.py" "$BUILD_DIR/"
if [ -f "$REPO_DIR/saved_model/config.json" ]; then
    echo "Dùng trọng số local: $REPO_DIR/saved_model"
    cp -r "$REPO_DIR/saved_model" "$BUILD_DIR/saved_model"
fi

echo "Deploy Grounding DINO Window ($MODE)..."
"$NUCTL" create project cvat --platform local >/dev/null 2>&1 || true
"$NUCTL" deploy --project-name cvat \
    --path "$BUILD_DIR" \
    --file "$BUILD_DIR/function.yaml" \
    --platform local \
    --platform-config "{\"attributes\": {\"network\": \"$CVAT_NETWORK\"}}"

"$NUCTL" get function --platform local
