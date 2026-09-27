#!/usr/bin/env bash
# Nâng cấp CVAT trên server theo phiên bản submodule cvat/ đã được commit trong repo.
# Chạy SAU khi `git pull`:
#
#   git pull && ./scripts/upgrade_cvat.sh
#   ./scripts/upgrade_cvat.sh --no-backup     # bỏ qua backup (không khuyến khích)
#
# Các bước: backup volume -> cập nhật submodule -> pull image -> khởi động (CVAT tự migrate DB)
# -> chờ CVAT sẵn sàng -> deploy lại model.
# Backup lưu ở backups/<thời gian>/, khôi phục theo
# https://docs.cvat.ai/docs/administration/community/advanced/backup_guide/

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CVAT="$ROOT_DIR/scripts/cvat.sh"

BACKUP=1
for arg in "$@"; do
    case "$arg" in
        --no-backup) BACKUP=0 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "Tham số không hợp lệ: $arg" >&2; exit 1 ;;
    esac
done

git -C "$ROOT_DIR" submodule update --init --depth 1 cvat
echo "CVAT source: $(git -C "$ROOT_DIR/cvat" describe --tags 2>/dev/null || git -C "$ROOT_DIR/cvat" rev-parse --short HEAD)"

if [ "$BACKUP" = 1 ] && docker inspect cvat_db >/dev/null 2>&1; then
    BACKUP_DIR="$ROOT_DIR/backups/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    echo "Dừng CVAT để backup vào $BACKUP_DIR ..."
    "$CVAT" stop

    backup() { # <container> <thư mục trong container> <tên file>
        docker run --rm --volumes-from "$1" -v "$BACKUP_DIR:/backup" ubuntu \
            tar -czf "/backup/$3" "$2"
    }
    backup cvat_db /var/lib/postgresql/data cvat_db.tar.gz
    backup cvat_server /home/django/data cvat_data.tar.gz
    backup cvat_clickhouse /var/lib/clickhouse cvat_events_db.tar.gz
    ls -lh "$BACKUP_DIR"
fi

"$CVAT" pull
"$CVAT" up -d

echo "Chờ CVAT khởi động (migrate DB có thể mất vài phút, xem: ./scripts/cvat.sh logs -f cvat_server) ..."
for _ in $(seq 1 120); do
    if docker exec cvat_server curl -sf -o /dev/null http://localhost:8080/api/server/about 2>/dev/null; then
        echo "CVAT đã sẵn sàng."
        break
    fi
    sleep 10
done
docker exec cvat_server curl -sf -o /dev/null http://localhost:8080/api/server/about \
    || { echo "CVAT chưa sẵn sàng sau 20 phút, kiểm tra log cvat_server" >&2; exit 1; }

# Nuclio có thể đã lên version mới -> build lại function bằng nuctl tương ứng
"$ROOT_DIR/scripts/deploy_models.sh"
