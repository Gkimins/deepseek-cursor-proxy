from __future__ import annotations

import json
import unittest

from deepseek_cursor_proxy.anthropic_transform import (
    RECOVERY_NOTICE_CONTENT,
    RECOVERY_NOTICE_TEXT,
    RECOVERY_SYSTEM_CONTENT,
    anthropic_cache_namespace,
    anthropic_canonical_scope_message,
    anthropic_conversation_scope,
    anthropic_message_signature,
    assistant_needs_thinking_for_tool_context,
    content_without_thinking,
    extract_thinking_from_content,
    has_anthropic_recovery_notice,
    has_thinking_blocks,
    has_tool_result_blocks,
    has_tool_use_blocks,
    lookup_anthropic_thinking,
    normalize_anthropic_content_blocks,
    normalize_anthropic_message,
    normalize_anthropic_messages,
    normalize_reasoning_effort,
    normalize_tool_use_block,
    prepare_anthropic_upstream_request,
    prefix_anthropic_response_content,
    record_anthropic_response_reasoning,
    recover_anthropic_messages,
    rewrite_anthropic_response_body,
    store_anthropic_assistant_reasoning,
    strip_cursor_thinking_blocks_from_text,
    tool_use_ids_from_content,
    tool_use_signature,
    upstream_model_for,
)
from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore


def _make_scope(messages: list[dict], namespace: str = "") -> str:
    return anthropic_conversation_scope(messages, namespace)


class AnthropicTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ReasoningStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    # --- normalize_reasoning_effort ---

    def test_normalize_reasoning_effort_defaults_to_high_for_non_string(self) -> None:
        self.assertEqual(normalize_reasoning_effort(None), "high")
        self.assertEqual(normalize_reasoning_effort(123), "high")

    def test_normalize_reasoning_effort_maps_aliases(self) -> None:
        self.assertEqual(normalize_reasoning_effort("low"), "high")
        self.assertEqual(normalize_reasoning_effort("medium"), "high")
        self.assertEqual(normalize_reasoning_effort("high"), "high")
        self.assertEqual(normalize_reasoning_effort("max"), "max")
        self.assertEqual(normalize_reasoning_effort("xhigh"), "max")

    def test_normalize_reasoning_effort_unknown_value_defaults_to_high(self) -> None:
        self.assertEqual(normalize_reasoning_effort("unknown"), "high")

    # --- extract_thinking_from_content ---

    def test_extract_thinking_returns_none_for_non_list(self) -> None:
        self.assertIsNone(extract_thinking_from_content("text"))
        self.assertIsNone(extract_thinking_from_content(None))

    def test_extract_thinking_concatenates_thinking_blocks(self) -> None:
        content = [
            {"type": "thinking", "thinking": "Step 1."},
            {"type": "text", "text": "Hello"},
            {"type": "thinking", "thinking": "Step 2."},
        ]
        self.assertEqual(extract_thinking_from_content(content), "Step 1.\nStep 2.")

    def test_extract_thinking_ignores_non_string_thinking(self) -> None:
        content = [{"type": "thinking", "thinking": ["not a string"]}]
        self.assertIsNone(extract_thinking_from_content(content))

    # --- content_without_thinking ---

    def test_content_without_thinking_filters_thinking_blocks(self) -> None:
        content = [
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": "Hi"},
            {"type": "tool_use", "name": "f", "input": {}},
        ]
        result = content_without_thinking(content)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["type"], "text")
        self.assertEqual(result[1]["type"], "tool_use")

    def test_content_without_thinking_returns_empty_for_non_list(self) -> None:
        self.assertEqual(content_without_thinking("text"), [])

    # --- block type checks ---

    def test_has_tool_use_blocks(self) -> None:
        self.assertTrue(has_tool_use_blocks([{"type": "tool_use", "name": "f"}]))
        self.assertFalse(has_tool_use_blocks([{"type": "text", "text": "x"}]))
        self.assertFalse(has_tool_use_blocks("not a list"))

    def test_has_thinking_blocks(self) -> None:
        self.assertTrue(has_thinking_blocks([{"type": "thinking", "thinking": "t"}]))
        self.assertFalse(has_thinking_blocks([{"type": "text", "text": "x"}]))
        self.assertFalse(has_thinking_blocks("not a list"))

    def test_has_tool_result_blocks(self) -> None:
        self.assertTrue(has_tool_result_blocks([{"type": "tool_result", "content": "c"}]))
        self.assertFalse(has_tool_result_blocks([{"type": "text", "text": "x"}]))
        self.assertFalse(has_tool_result_blocks("not a list"))

    # --- tool_use_ids_from_content ---

    def test_tool_use_ids_extracts_ids(self) -> None:
        content = [
            {"type": "tool_use", "id": "id1", "name": "a"},
            {"type": "tool_use", "id": "id2", "name": "b"},
        ]
        self.assertEqual(tool_use_ids_from_content(content), ["id1", "id2"])

    def test_tool_use_ids_skips_missing_ids(self) -> None:
        content = [
            {"type": "tool_use", "name": "a"},
            {"type": "tool_use", "id": "id1", "name": "b"},
        ]
        self.assertEqual(tool_use_ids_from_content(content), ["id1"])

    # --- normalize_tool_use_block ---

    def test_normalize_tool_use_block_parses_json_input(self) -> None:
        block = {"type": "tool_use", "name": "foo", "input": '{"a":1}'}
        self.assertEqual(
            normalize_tool_use_block(block),
            {"type": "tool_use", "name": "foo", "input": {"a": 1}},
        )

    def test_normalize_tool_use_block_handles_non_json_input(self) -> None:
        block = {"type": "tool_use", "name": "foo", "input": "not json"}
        self.assertEqual(
            normalize_tool_use_block(block),
            {"type": "tool_use", "name": "foo", "input": {}},
        )

    def test_normalize_tool_use_block_preserves_dict_input(self) -> None:
        block = {"type": "tool_use", "name": "foo", "input": {"a": 1}}
        self.assertEqual(normalize_tool_use_block(block)["input"], {"a": 1})

    # --- tool_use_signature ---

    def test_tool_use_signature_is_stable(self) -> None:
        block = {"type": "tool_use", "name": "foo", "input": {"a": 1}}
        sig1 = tool_use_signature(block)
        sig2 = tool_use_signature(block)
        self.assertEqual(sig1, sig2)

    def test_tool_use_signature_differs_for_different_inputs(self) -> None:
        sig_a = tool_use_signature({"type": "tool_use", "name": "f", "input": {"a": 1}})
        sig_b = tool_use_signature({"type": "tool_use", "name": "f", "input": {"b": 2}})
        self.assertNotEqual(sig_a, sig_b)

    # --- anthropic_message_signature ---

    def test_anthropic_message_signature_ignores_thinking_content(self) -> None:
        msg1_sig = anthropic_message_signature({
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi"}],
        })
        msg2_sig = anthropic_message_signature({
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "..."},
                {"type": "text", "text": "Hi"},
            ],
        })
        self.assertEqual(msg1_sig, msg2_sig)

    # --- anthropic_canonical_scope_message ---

    def test_canonical_scope_message_drops_thinking(self) -> None:
        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "..."},
                {"type": "text", "text": "Hi"},
            ],
        }
        canonical = anthropic_canonical_scope_message(msg)
        self.assertNotIn("thinking", str(canonical["content"]))

    # --- anthropic_conversation_scope ---

    def test_anthropic_conversation_scope_is_deterministic(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(
            anthropic_conversation_scope(messages),
            anthropic_conversation_scope(messages),
        )

    def test_anthropic_conversation_scope_differs_by_namespace(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        s1 = anthropic_conversation_scope(messages, "ns-a")
        s2 = anthropic_conversation_scope(messages, "ns-b")
        self.assertNotEqual(s1, s2)

    # --- store_anthropic_assistant_reasoning ---

    def test_store_anthropic_assistant_reasoning_ignores_non_assistant(self) -> None:
        self.assertEqual(
            store_anthropic_assistant_reasoning(
                self.store, {"role": "user", "content": "hi"}, "scope"
            ),
            0,
        )

    def test_store_anthropic_assistant_reasoning_stores_thinking(self) -> None:
        content = [
            {"type": "thinking", "thinking": "Let me think."},
            {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}},
        ]
        stored = store_anthropic_assistant_reasoning(
            self.store,
            {"role": "assistant", "content": content},
            "scope",
        )
        self.assertGreater(stored, 0)
        result = self.store.get("scope:scope:anthropic_tool_use:tu_1")
        self.assertEqual(result, "Let me think.")

    def test_store_anthropic_reasoning_no_thinking_returns_zero(self) -> None:
        stored = store_anthropic_assistant_reasoning(
            self.store,
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            "scope",
        )
        self.assertEqual(stored, 0)

    # --- lookup_anthropic_thinking ---

    def test_lookup_anthropic_thinking_by_signature(self) -> None:
        content = [
            {"type": "thinking", "thinking": "Plan."},
            {"type": "text", "text": "Answer."},
        ]
        store_anthropic_assistant_reasoning(
            self.store, {"role": "assistant", "content": content}, "scope"
        )
        result = lookup_anthropic_thinking(
            self.store,
            {"role": "assistant", "content": [{"type": "text", "text": "Answer."}]},
            "scope",
        )
        self.assertEqual(result, "Plan.")

    def test_lookup_anthropic_thinking_by_tool_use_id(self) -> None:
        content = [
            {"type": "thinking", "thinking": "Need tool."},
            {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}},
        ]
        store_anthropic_assistant_reasoning(
            self.store, {"role": "assistant", "content": content}, "scope"
        )
        # Lookup by different message but same tool_use id
        result = lookup_anthropic_thinking(
            self.store,
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Different text."},
                    {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}},
                ],
            },
            "scope",
        )
        self.assertEqual(result, "Need tool.")

    def test_lookup_anthropic_thinking_by_tool_use_signature(self) -> None:
        content = [
            {"type": "thinking", "thinking": "Using tool."},
            {"type": "tool_use", "name": "f", "input": {"a": 1}},
        ]
        store_anthropic_assistant_reasoning(
            self.store, {"role": "assistant", "content": content}, "scope"
        )
        result = lookup_anthropic_thinking(
            self.store,
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "f", "input": {"a": 1}}],
            },
            "scope",
        )
        self.assertEqual(result, "Using tool.")

    def test_lookup_anthropic_thinking_returns_none_for_no_match(self) -> None:
        result = lookup_anthropic_thinking(
            self.store,
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            "scope",
        )
        self.assertIsNone(result)

    # --- assistant_needs_thinking_for_tool_context ---

    def test_assistant_needs_thinking_when_has_tool_use(self) -> None:
        msg = {"role": "assistant", "content": [{"type": "tool_use", "name": "f"}]}
        self.assertTrue(assistant_needs_thinking_for_tool_context(msg, []))

    def test_assistant_needs_thinking_after_tool_result_in_user(self) -> None:
        prior = [
            {"role": "user", "content": [{"type": "tool_result", "content": "x"}]},
        ]
        msg = {"role": "assistant", "content": [{"type": "text", "text": "Answer."}]}
        self.assertTrue(assistant_needs_thinking_for_tool_context(msg, prior))

    def test_assistant_needs_thinking_plain_user_no_tool_result(self) -> None:
        prior = [{"role": "user", "content": "plain question"}]
        msg = {"role": "assistant", "content": [{"type": "text", "text": "Answer."}]}
        self.assertFalse(assistant_needs_thinking_for_tool_context(msg, prior))

    # --- strip_cursor_thinking_blocks_from_text ---

    def test_strip_cursor_thinking_blocks_removes_think_tags(self) -> None:
        text = "<think>\nReasoning.\n</think>\n\nFinal answer."
        self.assertEqual(strip_cursor_thinking_blocks_from_text(text), "Final answer.")

    def test_strip_cursor_thinking_blocks_handles_thinking_tag(self) -> None:
        text = "<thinking>\nReasoning.\n</thinking>\n\nDone."
        self.assertEqual(strip_cursor_thinking_blocks_from_text(text), "Done.")

    def test_strip_cursor_thinking_blocks_unclosed_tag(self) -> None:
        text = "<think>\nPartial reasoning.\nFinal."
        # The regex [\s\S]*? matches lazily to end-of-string ($) when no closing tag exists,
        # then \s* strips trailing whitespace — the entire string is consumed
        self.assertEqual(strip_cursor_thinking_blocks_from_text(text), "")

    # --- normalize_anthropic_content_blocks ---

    def test_normalize_anthropic_content_blocks_strips_thinking_from_text(self) -> None:
        result = normalize_anthropic_content_blocks(
            "Hello.\n<think>\nReasoning\n</think>\n\nWorld"
        )
        # strip_cursor_thinking_blocks_from_text does lstrip("\r\n") after regex removal
        self.assertEqual(result, "Hello.\nWorld")

    def test_normalize_anthropic_content_blocks_normalizes_list(self) -> None:
        content = [
            {"type": "text", "text": "<think>x</think>\n\nHi"},
            {"type": "thinking", "thinking": "keep"},
            {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
            {"type": "tool_result", "content": "r"},
        ]
        result = normalize_anthropic_content_blocks(content)
        self.assertEqual(result[0], {"type": "text", "text": "Hi"})
        self.assertEqual(result[1], {"type": "thinking", "thinking": "keep"})

    def test_normalize_anthropic_content_blocks_preserves_unknown_type(self) -> None:
        content = [{"type": "custom", "data": "x"}]
        result = normalize_anthropic_content_blocks(content)
        self.assertEqual(result[0]["type"], "custom")

    # --- normalize_anthropic_message ---

    def test_normalize_anthropic_message_converts_non_dict_to_user(self) -> None:
        result, patched, missing = normalize_anthropic_message(
            "plain text", None, [], "", True, True
        )
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["content"], "plain text")

    def test_normalize_anthropic_message_strips_thinking_from_string_content(self) -> None:
        result, _, _ = normalize_anthropic_message(
            {"role": "assistant", "content": "<think>x</think>\n\nHi"},
            None, [], "", False, True,
        )
        self.assertEqual(result["content"], "Hi")

    def test_normalize_anthropic_message_removes_thinking_when_keep_thinking_false(self) -> None:
        content = [
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": "Hi"},
        ]
        result, _, _ = normalize_anthropic_message(
            {"role": "assistant", "content": content},
            None, [], "", False, False,
        )
        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0]["type"], "text")

    def test_normalize_anthropic_message_repairs_thinking_from_cache(self) -> None:
        content = [
            {"type": "thinking", "thinking": "Need tool."},
            {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}},
        ]
        scope = anthropic_conversation_scope([], "test-ns")
        store_anthropic_assistant_reasoning(
            self.store, {"role": "assistant", "content": content}, scope
        )

        # Second message with same tool_use but no thinking block
        result, patched, missing = normalize_anthropic_message(
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "f", "input": {}}]},
            self.store, [], "test-ns", True, True,
        )
        self.assertTrue(patched)
        self.assertFalse(missing)
        self.assertEqual(result["content"][0]["type"], "thinking")
        self.assertEqual(result["content"][0]["thinking"], "Need tool.")

    def test_normalize_anthropic_message_reports_missing_when_no_cache(self) -> None:
        result, patched, missing = normalize_anthropic_message(
            {"role": "assistant", "content": [{"type": "tool_use", "name": "f"}]},
            self.store, [], "test-ns", True, True,
        )
        self.assertFalse(patched)
        self.assertTrue(missing)

    # --- normalize_anthropic_messages ---

    def test_normalize_anthropic_messages_returns_empty_for_non_list(self) -> None:
        msgs, count, missing = normalize_anthropic_messages(
            None, self.store, "", True, True
        )
        self.assertEqual(msgs, [])
        self.assertEqual(count, 0)
        self.assertEqual(missing, [])

    # --- has_anthropic_recovery_notice ---

    def test_has_anthropic_recovery_notice_in_string_content(self) -> None:
        msg = {"role": "assistant", "content": RECOVERY_NOTICE_TEXT + " rest of answer"}
        self.assertTrue(has_anthropic_recovery_notice(msg))

    def test_has_anthropic_recovery_notice_in_text_block(self) -> None:
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": RECOVERY_NOTICE_TEXT + " rest"}],
        }
        self.assertTrue(has_anthropic_recovery_notice(msg))

    def test_has_anthropic_recovery_notice_ignores_non_assistant(self) -> None:
        msg = {"role": "user", "content": RECOVERY_NOTICE_TEXT}
        self.assertFalse(has_anthropic_recovery_notice(msg))

    # --- recover_anthropic_messages ---

    def test_recover_anthropic_messages_from_last_user(self) -> None:
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": [{"type": "tool_use", "name": "f"}]},
        ]
        recovered, omitted, notice = recover_anthropic_messages(messages, [3])
        # Last user at index 2; returns [messages[2]] + recovery notice
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["role"], "user")
        self.assertEqual(recovered[0]["content"], "q2")
        self.assertEqual(notice, RECOVERY_NOTICE_CONTENT)
        self.assertGreater(omitted, 0)

    def test_recover_anthropic_messages_no_user_returns_all(self) -> None:
        messages = [
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": [{"type": "tool_use", "name": "f"}]},
        ]
        recovered, omitted, notice = recover_anthropic_messages(messages, [1])
        self.assertEqual(len(recovered), 2)
        self.assertEqual(omitted, 0)
        self.assertIsNone(notice)

    def test_recover_anthropic_messages_respects_boundary(self) -> None:
        """When a later message already has a recovery notice and earlier message is missing,
        the boundary triggers and drops everything before the recovery notice."""
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": RECOVERY_NOTICE_TEXT},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": [{"type": "tool_use", "name": "f"}]},
        ]
        # missing_index [1] (the old assistant at index 1) is before recovery notice at [3]
        recovered, omitted, notice = recover_anthropic_messages(messages, [1])
        self.assertIsNone(notice)

    # --- upstream_model_for ---

    def test_upstream_model_for_uses_config_model_for_non_deepseek(self) -> None:
        self.assertEqual(
            upstream_model_for("gpt-4", ProxyConfig(upstream_model="deepseek-v4-pro")),
            "deepseek-v4-pro",
        )

    def test_upstream_model_for_preserves_deepseek_prefix(self) -> None:
        self.assertEqual(
            upstream_model_for("deepseek-v4-flash", ProxyConfig()),
            "deepseek-v4-flash",
        )

    # --- anthropic_cache_namespace ---

    def test_anthropic_cache_namespace_includes_api_format(self) -> None:
        ns = anthropic_cache_namespace(ProxyConfig(), "deepseek-v4-pro", {"type": "enabled"}, "high")
        self.assertTrue(len(ns) > 0)

    def test_anthropic_cache_namespace_differs_by_auth(self) -> None:
        ns_a = anthropic_cache_namespace(ProxyConfig(), "m", {"type": "enabled"}, "high", "Bearer a")
        ns_b = anthropic_cache_namespace(ProxyConfig(), "m", {"type": "enabled"}, "high", "Bearer b")
        self.assertNotEqual(ns_a, ns_b)

    # --- prepare_anthropic_upstream_request ---

    def test_prepare_anthropic_upstream_request_basic(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
        prepared = prepare_anthropic_upstream_request(payload, ProxyConfig(), self.store)
        self.assertEqual(prepared.original_model, "deepseek-v4-pro")
        self.assertEqual(prepared.payload["model"], "deepseek-v4-pro")
        self.assertEqual(prepared.payload["messages"][0]["role"], "user")

    def test_prepare_anthropic_upstream_request_falls_back_to_config_model(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        prepared = prepare_anthropic_upstream_request(
            payload, ProxyConfig(upstream_model="deepseek-v4-flash"), self.store
        )
        self.assertEqual(prepared.upstream_model, "deepseek-v4-flash")

    def test_prepare_anthropic_upstream_request_filters_unknown_fields(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "unknown_field": "should_be_removed",
        }
        prepared = prepare_anthropic_upstream_request(payload, ProxyConfig(), self.store)
        self.assertNotIn("unknown_field", prepared.payload)

    def test_prepare_anthropic_upstream_request_enables_thinking(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "hi"}],
        }
        prepared = prepare_anthropic_upstream_request(
            payload, ProxyConfig(thinking="enabled"), self.store
        )
        self.assertEqual(prepared.payload["thinking"], {"type": "enabled"})
        self.assertEqual(prepared.payload["reasoning_effort"], "high")

    def test_prepare_anthropic_upstream_request_pass_through_thinking(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "disabled"},
        }
        prepared = prepare_anthropic_upstream_request(
            payload, ProxyConfig(thinking="pass-through"), self.store
        )
        self.assertEqual(prepared.payload["thinking"], {"type": "disabled"})

    def test_prepare_anthropic_upstream_request_injects_recovery_notice_into_system(self) -> None:
        """When missing reasoning occurs with recover strategy, notice is added to system."""
        payload = {
            "model": "deepseek-v4-pro",
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": [{"type": "tool_use", "name": "f"}]},
            ],
        }
        prepared = prepare_anthropic_upstream_request(
            payload,
            ProxyConfig(thinking="enabled", missing_reasoning_strategy="recover"),
            self.store,
        )
        self.assertIn(RECOVERY_SYSTEM_CONTENT, str(prepared.payload.get("system", "")))

    def test_prepare_anthropic_upstream_request_adds_recovery_system(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": [{"type": "tool_use", "name": "f"}]},
            ],
        }
        prepared = prepare_anthropic_upstream_request(
            payload,
            ProxyConfig(thinking="enabled", missing_reasoning_strategy="recover"),
            self.store,
        )
        self.assertEqual(prepared.payload["system"], RECOVERY_SYSTEM_CONTENT)

    # --- prefix_anthropic_response_content ---

    def test_prefix_anthropic_response_content_prepends_to_text_block(self) -> None:
        payload = {"content": [{"type": "text", "text": "Answer."}]}
        self.assertTrue(prefix_anthropic_response_content(payload, "[Prefix] "))
        self.assertEqual(payload["content"][0]["text"], "[Prefix] Answer.")

    def test_prefix_anthropic_response_content_inserts_text_block(self) -> None:
        payload = {"content": [{"type": "tool_use", "name": "f"}]}
        self.assertTrue(prefix_anthropic_response_content(payload, "[Prefix] "))
        self.assertEqual(payload["content"][0]["type"], "text")
        self.assertEqual(payload["content"][0]["text"], "[Prefix] ")

    def test_prefix_anthropic_response_content_non_list(self) -> None:
        self.assertFalse(prefix_anthropic_response_content({"content": "str"}, "[P]"))

    # --- record_anthropic_response_reasoning ---

    def test_record_anthropic_response_reasoning_stores_thinking(self) -> None:
        response = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me check."},
                {"type": "text", "text": "Done."},
            ],
        }
        count = record_anthropic_response_reasoning(
            response, self.store, [{"role": "user", "content": "hi"}]
        )
        self.assertGreater(count, 0)

    def test_record_anthropic_response_reasoning_null_store(self) -> None:
        self.assertEqual(
            record_anthropic_response_reasoning({"content": []}, None, []),
            0,
        )

    # --- rewrite_anthropic_response_body ---

    def test_rewrite_anthropic_response_body_restores_model(self) -> None:
        body = json.dumps({
            "id": "msg_test",
            "model": "deepseek-v4-pro",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi"}],
        }).encode()
        rewritten = rewrite_anthropic_response_body(body, "deepseek-v4-flash", self.store, [])
        payload = json.loads(rewritten)
        self.assertEqual(payload["model"], "deepseek-v4-flash")

    def test_rewrite_anthropic_response_body_prefixes_content(self) -> None:
        body = json.dumps({
            "id": "msg_test",
            "model": "deepseek-v4-pro",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi"}],
        }).encode()
        rewritten = rewrite_anthropic_response_body(
            body, "deepseek-v4-pro", self.store, [],
            content_prefix="[Recovered] ",
        )
        payload = json.loads(rewritten)
        self.assertTrue(payload["content"][0]["text"].startswith("[Recovered] "))

    # --- Conversation scope isolation ---

    def test_anthropic_cache_scoped_by_conversation(self) -> None:
        tool_use = {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}}
        content_a = [
            {"type": "thinking", "thinking": "Thread A."},
            tool_use,
        ]
        content_b = [
            {"type": "thinking", "thinking": "Thread B."},
            tool_use,
        ]
        scope_a = anthropic_conversation_scope([{"role": "user", "content": "thread A"}])
        scope_b = anthropic_conversation_scope([{"role": "user", "content": "thread B"}])
        store_anthropic_assistant_reasoning(
            self.store, {"role": "assistant", "content": content_a}, scope_a
        )
        store_anthropic_assistant_reasoning(
            self.store, {"role": "assistant", "content": content_b}, scope_b
        )
        result_a = lookup_anthropic_thinking(
            self.store,
            {"role": "assistant", "content": [tool_use]},
            scope_a,
        )
        result_b = lookup_anthropic_thinking(
            self.store,
            {"role": "assistant", "content": [tool_use]},
            scope_b,
        )
        self.assertEqual(result_a, "Thread A.")
        self.assertEqual(result_b, "Thread B.")


if __name__ == "__main__":
    unittest.main()
