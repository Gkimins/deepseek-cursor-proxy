from __future__ import annotations

import json
import unittest

from deepseek_cursor_proxy.anthropic_streaming import (
    AnthropicStreamAccumulator,
    ContentBlock,
    parse_anthropic_sse_events,
)
from deepseek_cursor_proxy.reasoning_store import ReasoningStore


class ContentBlockTests(unittest.TestCase):
    def test_to_block_thinking(self) -> None:
        block = ContentBlock(block_type="thinking", thinking="Steps.")
        self.assertEqual(block.to_block(), {"type": "thinking", "thinking": "Steps."})

    def test_to_block_text(self) -> None:
        block = ContentBlock(block_type="text", text="Hello")
        self.assertEqual(block.to_block(), {"type": "text", "text": "Hello"})

    def test_to_block_tool_use_with_valid_json(self) -> None:
        block = ContentBlock(
            block_type="tool_use",
            tool_use_id="tu_1",
            tool_use_name="read_file",
            tool_use_input_json='{"path":"README.md"}',
        )
        expected = {
            "type": "tool_use",
            "id": "tu_1",
            "name": "read_file",
            "input": {"path": "README.md"},
        }
        self.assertEqual(block.to_block(), expected)

    def test_to_block_tool_use_with_invalid_json(self) -> None:
        block = ContentBlock(
            block_type="tool_use",
            tool_use_name="bad_tool",
            tool_use_input_json="not valid json",
        )
        result = block.to_block()
        self.assertEqual(result["input"], "not valid json")

    def test_to_block_tool_use_without_id(self) -> None:
        block = ContentBlock(
            block_type="tool_use",
            tool_use_name="f",
            tool_use_input_json='{"a":1}',
        )
        result = block.to_block()
        self.assertNotIn("id", result)

    def test_to_block_unknown_type(self) -> None:
        block = ContentBlock(block_type="custom_type")
        self.assertEqual(block.to_block(), {"type": "custom_type"})


class AnthropicStreamAccumulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.acc = AnthropicStreamAccumulator()
        self.store = ReasoningStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_accumulator_builds_message_from_events(self) -> None:
        self.acc.ingest_event("message_start", {
            "message": {"id": "msg_1", "model": "deepseek-v4-pro", "usage": {"input_tokens": 10}},
        })
        self.acc.ingest_event("content_block_start", {
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello "},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 0,
            "delta": {"type": "text_delta", "text": "world."},
        })
        self.acc.ingest_event("content_block_stop", {"index": 0})
        self.acc.ingest_event("message_delta", {
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 5},
        })

        message = self.acc.to_message()
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(len(message["content"]), 1)
        self.assertEqual(message["content"][0]["type"], "text")
        self.assertEqual(message["content"][0]["text"], "Hello world.")
        self.assertEqual(self.acc.stop_reason, "end_turn")

    def test_accumulator_collects_thinking_and_tool_use(self) -> None:
        self.acc.ingest_event("message_start", {
            "message": {"id": "msg_1", "model": "deepseek-v4-pro"},
        })
        self.acc.ingest_event("content_block_start", {
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "I need a tool."},
        })
        self.acc.ingest_event("content_block_start", {
            "index": 1,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "read_file"},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"path":'},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '"README.md"}'},
        })

        message = self.acc.to_message()
        self.assertEqual(len(message["content"]), 2)
        self.assertEqual(message["content"][0]["type"], "thinking")
        self.assertEqual(message["content"][0]["thinking"], "I need a tool.")
        self.assertEqual(message["content"][1]["type"], "tool_use")
        self.assertEqual(message["content"][1]["input"], {"path": "README.md"})

    def test_store_reasoning_with_thinking_and_tool_use(self) -> None:
        self.acc.ingest_event("content_block_start", {
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Need tool."},
        })
        self.acc.ingest_event("content_block_start", {
            "index": 1,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "f"},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        })

        stored = self.acc.store_reasoning(self.store, "test-scope")
        self.assertGreater(stored, 0)

    def test_store_reasoning_no_thinking(self) -> None:
        self.acc.ingest_event("content_block_start", {
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 0,
            "delta": {"type": "text_delta", "text": "No thinking here."},
        })

        stored = self.acc.store_reasoning(self.store, "test-scope")
        self.assertEqual(stored, 0)

    def test_store_reasoning_idempotent(self) -> None:
        self.acc.ingest_event("content_block_start", {
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Test."},
        })
        self.acc.ingest_event("content_block_start", {
            "index": 1,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "f"},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        })

        first = self.acc.store_reasoning(self.store, "test-scope")
        second = self.acc.store_reasoning(self.store, "test-scope")
        self.assertGreater(first, 0)
        self.assertEqual(second, 0)

    def test_store_ready_reasoning_requires_tool_ids(self) -> None:
        self.acc.ingest_event("content_block_start", {
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Test."},
        })

        # No tool_use blocks → store_ready_reasoning returns 0
        stored = self.acc.store_ready_reasoning(self.store, "test-scope")
        self.assertEqual(stored, 0)

    def test_store_ready_reasoning_with_tool_ids(self) -> None:
        self.acc.ingest_event("content_block_start", {
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Need tool."},
        })
        self.acc.ingest_event("content_block_start", {
            "index": 1,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "f"},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        })

        stored = self.acc.store_ready_reasoning(self.store, "test-scope")
        self.assertGreater(stored, 0)

    def test_unknown_event_type_noop(self) -> None:
        self.acc.ingest_event("unknown_event", {"data": "x"})
        self.assertEqual(len(self.acc.blocks), 0)

    def test_blocks_sorted_by_index(self) -> None:
        self.acc.ingest_event("content_block_start", {
            "index": 2,
            "content_block": {"type": "text", "text": ""},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 2,
            "delta": {"type": "text_delta", "text": "C"},
        })
        self.acc.ingest_event("content_block_start", {
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 0,
            "delta": {"type": "text_delta", "text": "A"},
        })
        self.acc.ingest_event("content_block_start", {
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        })
        self.acc.ingest_event("content_block_delta", {
            "index": 1,
            "delta": {"type": "text_delta", "text": "B"},
        })

        message = self.acc.to_message()
        self.assertEqual(message["content"][0]["text"], "A")
        self.assertEqual(message["content"][1]["text"], "B")
        self.assertEqual(message["content"][2]["text"], "C")


class ParseAnthropicSSETests(unittest.TestCase):
    def test_parse_simple_event(self) -> None:
        lines = [
            b"event: message_start",
            b"data: {\"message\": {\"id\": \"msg_1\"}}",
            b"",
        ]
        events = parse_anthropic_sse_events(lines)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "message_start")
        self.assertEqual(events[0][1]["message"]["id"], "msg_1")

    def test_parse_multiple_events(self) -> None:
        lines = [
            b"event: content_block_start",
            b"data: {\"index\":0}",
            b"",
            b"event: content_block_delta",
            b"data: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}",
            b"",
            b"event: content_block_stop",
            b"data: {\"index\":0}",
            b"",
        ]
        events = parse_anthropic_sse_events(lines)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0][0], "content_block_start")
        self.assertEqual(events[1][0], "content_block_delta")
        self.assertEqual(events[2][0], "content_block_stop")

    def test_parse_invalid_json_becomes_empty_dict(self) -> None:
        lines = [
            b"data: not valid json",
            b"",
        ]
        events = parse_anthropic_sse_events(lines)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1], {})

    def test_parse_defaults_to_message_when_no_event_type(self) -> None:
        lines = [
            b"event: message_start",
            b"data: {}",
            b"",
            b"data: {}",
            b"",
        ]
        events = parse_anthropic_sse_events(lines)
        self.assertEqual(events[0][0], "message_start")
        # When no event: line, falls back to "message"
        self.assertEqual(events[1][0], "message")

    def test_parse_non_utf8_data(self) -> None:
        lines = [
            b"data: \xff\xfe",
            b"",
        ]
        events = parse_anthropic_sse_events(lines)
        self.assertEqual(events[0][1], {})

    def test_parse_empty_lines(self) -> None:
        self.assertEqual(parse_anthropic_sse_events([]), [])
        self.assertEqual(parse_anthropic_sse_events([b"", b""]), [])

    def test_parse_partial_event_no_blank_line(self) -> None:
        lines = [
            b"event: message_start",
            b"data: {\"key\": \"value\"}",
        ]
        # No trailing blank line → partial event not flushed
        events = parse_anthropic_sse_events(lines)
        self.assertEqual(len(events), 0)


if __name__ == "__main__":
    unittest.main()
