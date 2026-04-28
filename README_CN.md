<h1 align="center"><img src="assets/logo.png" width="150" alt="deepseek-cursor-proxy logo"><br>DeepSeek Cursor Proxy</h1>

<p align="center"><strong><a href="README.md">English</a> | <a href="README_CN.md">中文</a></strong></p>

一个兼容代理，通过正确处理 DeepSeek 工具调用推理 API 请求中的 `reasoning_content` 字段，将 Cursor 连接到 DeepSeek 推理模型（`deepseek-v4-pro` 和 `deepseek-v4-flash`）。

此代理同样适用于**遇到相同 `reasoning_content` 缺失问题的其他应用和编码代理**。只需将它们的 API base URL 指向该代理即可。

## 功能

- ✅ 在工具调用请求中注入 `reasoning_content`（因为 Cursor 不会包含该字段），从常规和流式 DeepSeek 响应中恢复先前缓存的推理内容。详见 [DeepSeek 文档](https://api-docs.deepseek.com/guides/thinking_mode#tool-calls)。
- ✅ 将流式 `reasoning_content` 镜像为 Cursor 可见的 `<think>...</think>` 文本块，使推理 token 显示在 Cursor UI 中。对于 BYOK 模式，Cursor 会将其渲染为普通文本。
- ✅ 启动 ngrok 隧道，使 Cursor 能通过公网 HTTPS URL 访问本地代理（默认关闭；使用 `--ngrok` 或 `ngrok: true` 启用）。
- ✅ 完整支持 Anthropic API 兼容（`/v1/messages` + SSE 流式传输）和 OpenAI 格式（`/v1/chat/completions`），两者均转发到 DeepSeek。
- ✅ 瞬时故障（HTTP 429、5xx）自动重试，指数退避（最多重试 3 次）。
- ✅ 提供其他兼容性修复，使 DeepSeek 模型在 Cursor 中良好运行。

## 为什么需要这个

此仓库修复启用推理模式时 Cursor + DeepSeek 工具调用的以下错误：

<img src="assets/error_400.png" width="600" alt="Error 400 - reasoning_content 必须回传">

```txt
⚠️ Connection Error
Provider returned error:
{
  "error": {
    "message": "The reasoning_content in the thinking mode must be passed back to the API.",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_request_error"
  }
}
```

## 使用方法

### 步骤 1：设置 ngrok

Cursor 会阻止 `localhost` 等非公网 API URL，因此代理需要一个公网 HTTPS URL。[ngrok](https://ngrok.com/) 可以将本地代理暴露给 Cursor，无需开放路由器端口。你也可以使用 [Cloudflare Tunnel](https://developers.cloudflare.com/tunnel/setup/)。创建一个 ngrok 账户并访问 [ngrok 控制台](https://dashboard.ngrok.com)，即可找到 authtoken 和公网 URL。

如果将此代理用于允许 localhost API 端点的其他应用，可以跳过此步骤——ngrok 默认关闭（自 v0.2 起）。如需启用，启动代理时加上 `--ngrok`，或在 `~/.deepseek-cursor-proxy/config.yaml` 中设置 `ngrok: true`。

<img src="assets/ngrok_dashboard.png" width="600" alt="ngrok 控制台">

然后一次性安装并认证 ngrok：

```bash
brew install ngrok
ngrok config add-authtoken <your-ngrok-token>
```

### 步骤 2：添加 Cursor 自定义模型

在 Cursor 中添加 DeepSeek 自定义模型并指向此代理：

- 模型：`deepseek-v4-pro`
- API Key：你的 DeepSeek API key
- Base URL：你的 ngrok HTTPS URL，末尾加上 `/v1` API 版本路径

代理会尊重 Cursor 发送的 DeepSeek 模型名称，如 `deepseek-v4-pro` 或 `deepseek-v4-flash`。`config.yaml` 中的 `model` 字段仅在请求未包含模型时作为回退使用。

例如，如果 ngrok 控制台显示 `https://example.ngrok-free.dev`，则使用：

```text
https://example.ngrok-free.dev/v1
```

<img src="assets/cursor_config.png" width="600" alt="Cursor 中 DeepSeek 代理设置">

快捷键切换自定义 API：

- macOS：`Cmd+Shift+0`
- Windows/Linux：`Ctrl+Shift+0`

### 步骤 3：安装并启动代理服务器

**使用 UV 运行**

```bash
# 如尚未安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装并启动
# uv 会将程序安装在仓库本地文件夹的 .venv/ 下
git clone https://github.com/yxlao/deepseek-cursor-proxy.git
cd deepseek-cursor-proxy
uv run deepseek-cursor-proxy
```

**使用 Conda 运行**

```bash
# 如尚未安装 conda
# 参考：https://www.anaconda.com/docs/getting-started/miniconda/install/overview

# 安装
conda create -n dcp python=3.10 -y
conda activate dcp
git clone https://github.com/yxlao/deepseek-cursor-proxy.git
cd deepseek-cursor-proxy
pip install -e .

# 启动
deepseek-cursor-proxy
```

如果启用了 ngrok，`deepseek-cursor-proxy` 启动时会打印 ngrok 公网 URL。如果与 Cursor 中设置的不同，请在 Cursor 的 Base URL 字段中更新。

首次运行时，`deepseek-cursor-proxy` 会创建：

- `~/.deepseek-cursor-proxy/config.yaml`：配置文件
- `~/.deepseek-cursor-proxy/reasoning_content.sqlite3`：推理内容缓存

持久化设置保存在 `~/.deepseek-cursor-proxy/config.yaml` 中。命令行参数会覆盖单次运行的配置，例如 `--ngrok`、`--port 9000` 或 `--verbose`。

代理支持将 OpenAI 格式（`/v1/chat/completions`）和 Anthropic 格式（`/v1/messages`）的请求转发到 DeepSeek。默认情况下，两者使用相同的 `base_url`。如需将 Anthropic 请求路由到不同的端点，可在配置文件中设置 `anthropic_base_url`，或通过命令行传递 `--anthropic-base-url`。

```yaml
# 示例：将 Anthropic 请求路由到不同端点
base_url: https://api.deepseek.com
anthropic_base_url: https://another-endpoint.example.com
```

### 步骤 4：在 Cursor 中与 DeepSeek 对话

在 Cursor 中选择 `deepseek-v4-pro`，像平常一样使用聊天或 Agent 模式。

<img src="assets/cursor_chat.png" width="480" alt="在 Cursor 中与 DeepSeek 聊天">

## Docker 部署

也可以使用 Docker 运行代理。配置文件与推理缓存通过卷挂载持久化到宿主机的 `~/.deepseek-cursor-proxy`。

### 使用 Docker

```bash
# 构建镜像
docker build -t deepseek-cursor-proxy .

# 运行容器
docker run -d \
  --name deepseek-cursor-proxy \
  -p 9000:9000 \
  -v ~/.deepseek-cursor-proxy:/root/.deepseek-cursor-proxy \
  deepseek-cursor-proxy

# 添加额外 CLI 参数
docker run -d \
  --name deepseek-cursor-proxy \
  -p 9000:9000 \
  -v ~/.deepseek-cursor-proxy:/root/.deepseek-cursor-proxy \
  deepseek-cursor-proxy --ngrok --verbose
```

### 使用 Docker Compose

```bash
docker compose up -d
```

编辑 `docker-compose.yml` 添加额外 CLI 参数或修改端口映射。

## 工作原理

- **核心修复：** DeepSeek 的[推理模式](https://api-docs.deepseek.com/guides/thinking_mode#tool-calls)要求后续请求必须回传 assistant 工具调用消息中的 `reasoning_content`，但 Cursor 会忽略该字段，导致 400 错误。代理（`Cursor → ngrok → proxy → DeepSeek API`）将每次 DeepSeek 响应中的 `reasoning_content` 存储在本地 SQLite 缓存中，按消息签名、工具调用 ID 和工具调用函数签名建立索引，并在请求到达 DeepSeek 之前补全缺失的 `reasoning_content`。冷缓存时（代理重启、模型切换），会记录并丢弃无法恢复的历史记录，从最新的用户请求继续，并在下一个 Cursor 响应前添加提示。
- **多对话隔离：** 为避免并发对话间的冲突，代理通过对规范化对话前缀（角色、内容和工具调用，不含 `reasoning_content`）的 SHA-256 哈希以及上游模型、配置和 API key 哈希来限定缓存键范围。不同线程获得不同的作用域，因此重复使用的工具调用 ID 不会冲突。字节级相同的克隆历史会产生相同的作用域。
- **上下文缓存兼容性：** 代理通过不注入合成线程 ID、时间戳或缓存控制消息来保持兼容性。它将 `reasoning_content` 恢复为完全相同的原始字符串，使得重复前缀保持完整，兼容 [DeepSeek 上下文缓存](https://api-docs.deepseek.com/guides/kv_cache)。缓存命中率在终端输出中记录。
- **额外兼容性修复：** 除推理修复外，代理还会将旧版 `functions`/`function_call` 字段转换为 `tools`/`tool_choice`，保留 required 和命名 tool-choice 语义，规范化 `reasoning_effort` 别名，从 assistant 内容中剥离镜像的 `<think>` 块，将多部分内容数组展平为纯文本，并将 `reasoning_content` 镜像为 Cursor 可见的 `<think>...</think>` 块。
- **Anthropic API 兼容：** 代理在 `/v1/messages` 接受 Anthropic 格式的请求，并将其转换到 DeepSeek 的原生 Anthropic 兼容端点。它处理 Anthropic SSE 流事件（`message_start`、`content_block_start/delta/stop`、`message_delta`、`message_stop`），使用相同的 SQLite 存储缓存和注入 thinking 内容（等效于 OpenAI 路径中的 `reasoning_content`），并使用 Anthropic 特定的缓存键命名空间，同时支持从缺失 thinking 的历史中恢复。
- **上游重试与退避：** 代理在瞬时故障时最多重试上游 API 请求 3 次——包括 HTTP 429（速率限制）、5xx 服务端错误和网络错误——采用指数退避延迟（1 秒、2 秒）。客户端错误（4xx，429 除外）立即失败不重试。OpenAI 和 Anthropic 路径共享此重试逻辑。

## 开发

运行单元测试：

```bash
uv run python -m unittest discover -s tests
```

运行 pre-commit 钩子（代码格式化和检查）：

```bash
uv sync --dev
uv run pre-commit run --all-files
```

## 调试

使用详细输出运行：

```bash
deepseek-cursor-proxy --verbose
```

不使用 ngrok 运行（ngrok 默认关闭），方便本地 curl 测试：

```bash
deepseek-cursor-proxy --port 9000 --verbose
```

使用其他配置文件：

```bash
deepseek-cursor-proxy --config ./dev.config.yaml
```

清除本地推理缓存：

```bash
deepseek-cursor-proxy --clear-reasoning-cache
```

将 Anthropic 请求路由到不同端点：

```bash
deepseek-cursor-proxy --anthropic-base-url https://custom-api.example.com
```
