#!/usr/bin/env bash
# Chạy docker compose của CVAT kèm serverless (Nuclio).
#
#   ./scripts/cvat.sh                # = up -d
#   ./scripts/cvat.sh ps
#   ./scripts/cvat.sh logs -f cvat_server
#   ./scripts/cvat.sh down
#
# Cấu hình đọc từ file .env ở thư mục gốc repo (xem .env.example).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT_DIR/.env"
    set +a
fi

if [ ! -f "$ROOT_DIR/cvat/docker-compose.yml" ]; then
    echo "Chưa có source CVAT, tải submodule..."
    git -C "$ROOT_DIR" submodule update --init --depth 1 cvat
fi

[ $# -eq 0 ] && set -- up -d

cd "$ROOT_DIR/cvat"
exec docker compose \
    -f docker-compose.yml \
    -f components/serverless/docker-compose.serverless.yml \
    "$@"
