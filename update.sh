#!/usr/bin/env bash
# deepseek-cursor-proxy 更新脚本
# 拉取最新代码,重建镜像并重启服务

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
SERVICE_NAME="deepseek-cursor-proxy"

echo "=== deepseek-cursor-proxy 更新脚本 ==="
echo ""

# 保存更新前的 commit hash
OLD_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "[info] 当前版本: ${OLD_COMMIT}"

# 拉取最新代码
echo "[info] 拉取最新代码..."
git pull --ff-only

NEW_COMMIT=$(git rev-parse --short HEAD)
echo "[info] 最新版本: ${NEW_COMMIT}"

if [ "$OLD_COMMIT" = "$NEW_COMMIT" ] && docker ps --format '{{.Names}}' | grep -q "^${SERVICE_NAME}$"; then
    echo "[info] 已是最新版本且服务正在运行，无需更新"
    docker compose -f "$COMPOSE_FILE" ps
    exit 0
fi

# 停止并移除旧容器
if docker ps -a --format '{{.Names}}' | grep -q "^${SERVICE_NAME}$"; then
    echo "[info] 停止旧容器..."
    docker compose -f "$COMPOSE_FILE" down
fi

# 重建镜像（无缓存，确保依赖更新）
echo "[info] 重建镜像..."
docker compose -f "$COMPOSE_FILE" build --no-cache

# 启动新容器
echo "[info] 启动新服务..."
docker compose -f "$COMPOSE_FILE" up -d

# 等待服务就绪
echo "[info] 等待服务就绪..."
for i in $(seq 1 15); do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:9000/healthz" 2>/dev/null | grep -q "200"; then
        echo "[ok] 更新完成: ${OLD_COMMIT} -> ${NEW_COMMIT}"
        echo "    服务地址: http://localhost:9000"
        echo "    Cursor Base URL: http://localhost:9000/v1"
        echo ""
        docker compose -f "$COMPOSE_FILE" ps
        exit 0
    fi
    sleep 1
done

echo "[warn] 服务可能仍在启动中，请检查日志:"
echo "  docker compose -f ${COMPOSE_FILE} logs -f"
