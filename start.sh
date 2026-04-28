#!/usr/bin/env bash
# deepseek-cursor-proxy 快速启动脚本
# 构建本地 Docker 镜像并启动服务

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
SERVICE_NAME="deepseek-cursor-proxy"

echo "=== deepseek-cursor-proxy 启动脚本 ==="
echo ""

# 确保 config 目录存在
CONFIG_DIR="${HOME}/.deepseek-cursor-proxy"
if [ ! -d "$CONFIG_DIR" ]; then
    echo "[info] 创建配置目录: ${CONFIG_DIR}"
    mkdir -p "$CONFIG_DIR"
fi

# 如果容器已存在则先停止并移除
if docker ps -a --format '{{.Names}}' | grep -q "^${SERVICE_NAME}$"; then
    echo "[info] 停止已有容器..."
    docker compose -f "$COMPOSE_FILE" down
fi

# 构建镜像
echo "[info] 构建镜像..."
docker compose -f "$COMPOSE_FILE" build

# 启动服务
echo "[info] 启动服务..."
docker compose -f "$COMPOSE_FILE" up -d

# 等待服务就绪
echo "[info] 等待服务就绪..."
for i in $(seq 1 15); do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:9000/healthz" 2>/dev/null | grep -q "200"; then
        echo "[ok] 服务已启动 http://localhost:9000"
        echo ""
        echo "Cursor 配置 Base URL: http://localhost:9000/v1"
        echo ""
        docker compose -f "$COMPOSE_FILE" ps
        exit 0
    fi
    sleep 1
done

echo "[warn] 服务可能仍在启动中，请检查日志:"
echo "  docker compose -f ${COMPOSE_FILE} logs -f"
