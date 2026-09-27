#!/usr/bin/env bash
# Build & deploy tất cả Nuclio function trong models/*/cvat_integration lên CVAT.
#
#   ./scripts/deploy_models.sh           # tự chọn GPU/CPU
#   ./scripts/deploy_models.sh --cpu
#   ./scripts/deploy_models.sh --gpu
#
# CVAT (kèm serverless) phải đang chạy: ./scripts/cvat.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT_DIR/.env"
    set +a
fi

failed=()
for script in "$ROOT_DIR"/models/*/cvat_integration/deploy_nuclio.sh; do
    model="$(basename "$(dirname "$(dirname "$script")")")"
    echo "=============== $model ==============="
    if ! (cd "$(dirname "$(dirname "$script")")" && "$script" "$@"); then
        failed+=("$model")
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "Deploy lỗi: ${failed[*]}" >&2
    exit 1
fi
echo "Đã deploy xong tất cả model."
