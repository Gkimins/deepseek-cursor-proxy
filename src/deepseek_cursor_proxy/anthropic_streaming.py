from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .anthropic_transform import (
    anthropic_conversation_scope,
    extract_thinking_from_content,
    store_anthropic_assistant_reasoning,
    tool_use_ids_from_content,
)
from .reasoning_store import ReasoningStore


@dataclass
class ContentBlock:
    block_type: str = ""
    index: int = 0
    thinking: str = ""
    text: str = ""
    tool_use_id: str = ""
    tool_use_name: str = ""
    tool_use_input_json: str = ""

    def to_block(self) -> dict[str, Any]:
        if self.block_type == "thinking":
            return {"type": "thinking", "thinking": self.thinking}
        if self.block_type == "text":
            return {"type": "text", "text": self.text}
        if self.block_type == "tool_use":
            inp: Any = {}
            if self.tool_use_input_json:
                try:
                    inp = json.loads(self.tool_use_input_json)
                except (json.JSONDecodeError, ValueError):
                    inp = self.tool_use_input_json
            result: dict[str, Any] = {
                "type": "tool_use",
                "name": self.tool_use_name,
                "input": inp,
            }
            if self.tool_use_id:
                result["id"] = self.tool_use_id
            return result
        return {"type": self.block_type}


class AnthropicStreamAccumulator:
    def __init__(self) -> None:
        self.blocks: dict[int, ContentBlock] = {}
        self.message_id: str = ""
        self.model: str = ""
        self.stop_reason: str | None = None
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self._stored: bool = False

    def ingest_event(self, event_type: str, data: dict[str, Any]) -> None:
        handler = _EVENT_HANDLERS.get(event_type)
        if handler is not None:
            handler(self, data)

    def to_message(self) -> dict[str, Any]:
        content = [
            self.blocks[i].to_block() for i in sorted(self.blocks)
        ]
        return {"role": "assistant", "content": content}

    def store_reasoning(self, store: ReasoningStore, scope: str) -> int:
        if self._stored:
            return 0
        message = self.to_message()
        thinking = extract_thinking_from_content(message.get("content"))
        if thinking is None:
            return 0
        stored = store_anthropic_assistant_reasoning(store, message, scope)
        if stored:
            self._stored = True
        return stored

    def store_ready_reasoning(self, store: ReasoningStore, scope: str) -> int:
        if self._stored:
            return 0
        message = self.to_message()
        content = message.get("content")
        thinking = extract_thinking_from_content(content)
        if thinking is None:
            return 0
        tool_ids = tool_use_ids_from_content(content)
        if not tool_ids:
            return 0
        if not all(tool_ids):
            return 0
        stored = store_anthropic_assistant_reasoning(store, message, scope)
        if stored:
            self._stored = True
        return stored

    def _handle_message_start(self, data: dict[str, Any]) -> None:
        message = data.get("message")
        if isinstance(message, dict):
            self.message_id = message.get("id", "")
            self.model = message.get("model", "")
            usage = message.get("usage")
            if isinstance(usage, dict):
                self.input_tokens = usage.get("input_tokens", 0)

    def _handle_content_block_start(self, data: dict[str, Any]) -> None:
        index = data.get("index", 0)
        block_data = data.get("content_block", {})
        if not isinstance(block_data, dict):
            block_data = {}
        block = ContentBlock(
            block_type=block_data.get("type", ""),
            index=index,
        )
        if block.block_type == "tool_use":
            block.tool_use_id = block_data.get("id", "")
            block.tool_use_name = block_data.get("name", "")
        self.blocks[index] = block

    def _handle_content_block_delta(self, data: dict[str, Any]) -> None:
        index = data.get("index", 0)
        block = self.blocks.get(index)
        if block is None:
            return
        delta = data.get("delta", {})
        if not isinstance(delta, dict):
            return
        delta_type = delta.get("type", "")
        if delta_type == "thinking_delta":
            block.thinking += delta.get("thinking", "")
        elif delta_type == "text_delta":
            block.text += delta.get("text", "")
        elif delta_type == "input_json_delta":
            block.tool_use_input_json += delta.get("partial_json", "")

    def _handle_content_block_stop(self, data: dict[str, Any]) -> None:
        pass

    def _handle_message_delta(self, data: dict[str, Any]) -> None:
        delta = data.get("delta", {})
        if isinstance(delta, dict):
            stop_reason = delta.get("stop_reason")
            if isinstance(stop_reason, str):
                self.stop_reason = stop_reason
        usage = data.get("usage", {})
        if isinstance(usage, dict):
            self.output_tokens = usage.get("output_tokens", self.output_tokens)

    def _handle_message_stop(self, data: dict[str, Any]) -> None:
        pass


_EVENT_HANDLERS = {
    "message_start": AnthropicStreamAccumulator._handle_message_start,
    "content_block_start": AnthropicStreamAccumulator._handle_content_block_start,
    "content_block_delta": AnthropicStreamAccumulator._handle_content_block_delta,
    "content_block_stop": AnthropicStreamAccumulator._handle_content_block_stop,
    "message_delta": AnthropicStreamAccumulator._handle_message_delta,
    "message_stop": AnthropicStreamAccumulator._handle_message_stop,
}


def parse_anthropic_sse_events(
    raw_lines: list[bytes],
) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    current_event = ""
    current_data = b""
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            if current_data:
                try:
                    data = json.loads(current_data.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    data = {}
                events.append((current_event or "message", data))
            current_event = ""
            current_data = b""
            continue
        if stripped.startswith(b"event:"):
            current_event = stripped[len(b"event:"):].strip().decode(
                "utf-8", errors="replace"
            )
        elif stripped.startswith(b"data:"):
            current_data = stripped[len(b"data:"):].strip()
    return events
