from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .config import ProxyConfig
from .reasoning_store import ReasoningStore


ANTHROPIC_SUPPORTED_REQUEST_FIELDS = {
    "model",
    "messages",
    "system",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "stream",
    "tools",
    "tool_choice",
    "thinking",
    "reasoning_effort",
    "metadata",
}

RECOVERY_NOTICE_TEXT = (
    "[deepseek-cursor-proxy] Recovered this DeepSeek chat because older "
    "tool-call reasoning was unavailable; continuing with recent context only."
)
RECOVERY_NOTICE_CONTENT = f"{RECOVERY_NOTICE_TEXT}\n\n"
RECOVERY_SYSTEM_CONTENT = (
    "deepseek-cursor-proxy recovered this request because older DeepSeek "
    "thinking-mode tool-call reasoning was unavailable. Older "
    "unrecoverable tool-call history was omitted; continue using only the "
    "remaining recovered context."
)

CURSOR_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking)>[\s\S]*?(?:</(?:think|thinking)>|$)\s*",
    re.IGNORECASE,
)

EFFORT_ALIASES = {
    "low": "high",
    "medium": "high",
    "high": "high",
    "max": "max",
    "xhigh": "max",
}


@dataclass(frozen=True)
class PreparedAnthropicRequest:
    payload: dict[str, Any]
    original_model: str
    upstream_model: str
    cache_namespace: str
    patched_thinking_messages: int
    missing_thinking_messages: int
    recovered_thinking_messages: int = 0
    recovery_dropped_messages: int = 0
    recovery_notice: str | None = None


def normalize_reasoning_effort(value: Any) -> str:
    if not isinstance(value, str):
        return "high"
    return EFFORT_ALIASES.get(value.strip().lower(), "high")


def extract_thinking_from_content(content: Any) -> str | None:
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "thinking":
            text = block.get("thinking", "")
            if isinstance(text, str):
                parts.append(text)
    if not parts:
        return None
    return "\n".join(parts)


def content_without_thinking(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [
        block
        for block in content
        if not (isinstance(block, dict) and block.get("type") == "thinking")
    ]


def has_tool_use_blocks(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_use"
        for block in content
    )


def has_thinking_blocks(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "thinking"
        for block in content
    )


def has_tool_result_blocks(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def tool_use_ids_from_content(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    return [
        str(block["id"])
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_use"
        and block.get("id")
    ]


def normalize_tool_use_block(block: dict[str, Any]) -> dict[str, Any]:
    inp = block.get("input", {})
    if not isinstance(inp, (dict, list)):
        try:
            inp = json.loads(str(inp))
        except (json.JSONDecodeError, TypeError, ValueError):
            inp = {}
    return {
        "type": "tool_use",
        "name": str(block.get("name", "")),
        "input": inp,
    }


def tool_use_signature(block: dict[str, Any]) -> str:
    normalized = normalize_tool_use_block(block)
    canonical = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def anthropic_message_signature(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        sig_content = content_without_thinking(content)
    elif isinstance(content, str):
        sig_content = content
    else:
        sig_content = ""
    payload = {"content": sig_content}
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def anthropic_canonical_scope_message(message: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {"role": message.get("role")}
    content = message.get("content")
    if isinstance(content, list):
        canonical["content"] = content_without_thinking(content)
    elif content is not None:
        canonical["content"] = content
    return canonical


def anthropic_conversation_scope(
    messages: list[dict[str, Any]], namespace: str = ""
) -> str:
    scope_messages = [anthropic_canonical_scope_message(m) for m in messages]
    payload: Any = scope_messages
    if namespace:
        payload = {"namespace": namespace, "messages": scope_messages}
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def store_anthropic_assistant_reasoning(
    store: ReasoningStore, message: dict[str, Any], scope: str
) -> int:
    if message.get("role") != "assistant":
        return 0
    content = message.get("content")
    thinking = extract_thinking_from_content(content)
    if thinking is None:
        return 0

    keys = [
        f"scope:{scope}:anthropic_sig:{anthropic_message_signature(message)}"
    ]
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_id = block.get("id")
                if tool_id:
                    keys.append(f"scope:{scope}:anthropic_tool_use:{tool_id}")
                keys.append(
                    f"scope:{scope}:anthropic_tool_use_sig:{tool_use_signature(block)}"
                )

    stored_message = {"role": "assistant", "content": content}
    for key in keys:
        store.put(key, thinking, stored_message)
    return len(keys)


def lookup_anthropic_thinking(
    store: ReasoningStore, message: dict[str, Any], scope: str
) -> str | None:
    result = store.get(
        f"scope:{scope}:anthropic_sig:{anthropic_message_signature(message)}"
    )
    if result is not None:
        return result

    content = message.get("content")
    for tool_id in tool_use_ids_from_content(content):
        result = store.get(f"scope:{scope}:anthropic_tool_use:{tool_id}")
        if result is not None:
            return result

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                result = store.get(
                    f"scope:{scope}:anthropic_tool_use_sig:{tool_use_signature(block)}"
                )
                if result is not None:
                    return result

    return None


def assistant_needs_thinking_for_tool_context(
    message: dict[str, Any],
    prior_messages: list[dict[str, Any]],
) -> bool:
    content = message.get("content")
    if has_tool_use_blocks(content):
        return True
    for prior in reversed(prior_messages):
        role = prior.get("role")
        if role == "user":
            prior_content = prior.get("content")
            if has_tool_result_blocks(prior_content):
                return True
            return False
        if role == "assistant":
            return False
    return False


def strip_cursor_thinking_blocks_from_text(text: str) -> str:
    return CURSOR_THINKING_BLOCK_RE.sub("", text).lstrip("\r\n")


def normalize_anthropic_content_blocks(content: Any) -> Any:
    if isinstance(content, str):
        stripped = strip_cursor_thinking_blocks_from_text(content)
        return stripped if stripped else ""
    if not isinstance(content, list):
        return content
    normalized: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if isinstance(text, str):
                text = strip_cursor_thinking_blocks_from_text(text)
            normalized.append({"type": "text", "text": text})
        elif block_type == "thinking":
            normalized.append(block)
        elif block_type == "tool_use":
            normalized.append(block)
        elif block_type == "tool_result":
            normalized.append(block)
        else:
            normalized.append(block)
    return normalized


def normalize_anthropic_message(
    message: Any,
    store: ReasoningStore | None,
    prior_messages: list[dict[str, Any]],
    cache_namespace: str,
    repair_thinking: bool,
    keep_thinking: bool,
) -> tuple[dict[str, Any], bool, bool]:
    if not isinstance(message, dict):
        return {"role": "user", "content": str(message)}, False, False

    normalized = dict(message)
    role = normalized.get("role", "user")

    if role == "assistant":
        content = normalized.get("content")
        if isinstance(content, str):
            normalized["content"] = strip_cursor_thinking_blocks_from_text(content)
        elif isinstance(content, list):
            normalized["content"] = normalize_anthropic_content_blocks(content)

    patched = False
    missing = False

    if role == "assistant":
        content = normalized.get("content")
        if not keep_thinking:
            if isinstance(content, list):
                normalized["content"] = content_without_thinking(content)
        elif repair_thinking:
            if not has_thinking_blocks(content):
                needs = assistant_needs_thinking_for_tool_context(
                    normalized, prior_messages
                )
                if needs and store is not None:
                    scope = anthropic_conversation_scope(
                        prior_messages, cache_namespace
                    )
                    restored = lookup_anthropic_thinking(
                        store, normalized, scope
                    )
                    if restored is not None:
                        thinking_block = {
                            "type": "thinking",
                            "thinking": restored,
                        }
                        existing = normalized.get("content")
                        if isinstance(existing, list):
                            normalized["content"] = [thinking_block] + list(
                                existing
                            )
                        elif isinstance(existing, str):
                            normalized["content"] = [
                                thinking_block,
                                {"type": "text", "text": existing},
                            ]
                        else:
                            normalized["content"] = [thinking_block]
                        patched = True
                if needs and not patched:
                    missing = True

    return normalized, patched, missing


def normalize_anthropic_messages(
    messages: Any,
    store: ReasoningStore | None,
    cache_namespace: str,
    repair_thinking: bool,
    keep_thinking: bool,
) -> tuple[list[dict[str, Any]], int, list[int]]:
    if not isinstance(messages, list):
        return [], 0, []
    normalized_list: list[dict[str, Any]] = []
    patched_count = 0
    missing_indexes: list[int] = []
    for msg in messages:
        norm, patched, missing_flag = normalize_anthropic_message(
            msg,
            store,
            normalized_list,
            cache_namespace,
            repair_thinking,
            keep_thinking,
        )
        normalized_list.append(norm)
        if patched:
            patched_count += 1
        if missing_flag:
            missing_indexes.append(len(normalized_list) - 1)
    return normalized_list, patched_count, missing_indexes


def has_anthropic_recovery_notice(message: dict[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    content = message.get("content")
    if isinstance(content, str):
        return content.startswith(RECOVERY_NOTICE_TEXT)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str) and text.startswith(
                    RECOVERY_NOTICE_TEXT
                ):
                    return True
    return False


def _preceding_assistant_for_tool_result(
    messages: list[dict[str, Any]],
    user_index: int,
) -> dict[str, Any] | None:
    """Return the assistant message immediately before *user_index* if the
    user message contains ``tool_result`` blocks that reference ``tool_use``
    blocks in that assistant message.  Keeping the pair together avoids
    Anthropic API 400 errors about orphaned ``tool_use_id`` references."""
    if user_index <= 0:
        return None
    user_msg = messages[user_index]
    if not has_tool_result_blocks(user_msg.get("content")):
        return None
    prev = messages[user_index - 1]
    if prev.get("role") == "assistant" and has_tool_use_blocks(prev.get("content")):
        return prev
    return None


def recover_anthropic_messages(
    messages: list[dict[str, Any]],
    missing_indexes: list[int],
) -> tuple[list[dict[str, Any]], int, str | None]:
    recovery_boundary = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if has_anthropic_recovery_notice(messages[i])
            and any(mi < i for mi in missing_indexes)
        ),
        -1,
    )
    if recovery_boundary != -1:
        context_user = next(
            (
                i
                for i in range(recovery_boundary - 1, -1, -1)
                if messages[i].get("role") == "user"
            ),
            -1,
        )
        recovered_tail = []
        if context_user != -1:
            prev_assistant = _preceding_assistant_for_tool_result(
                messages, context_user
            )
            if prev_assistant is not None:
                recovered_tail.append(prev_assistant)
            recovered_tail.append(messages[context_user])
        recovered_tail.extend(messages[recovery_boundary:])
        omitted = len(messages) - len(recovered_tail)
        return recovered_tail, omitted, None

    last_user = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if messages[i].get("role") == "user"
        ),
        -1,
    )
    if last_user == -1:
        return messages, 0, None

    recovered: list[dict[str, Any]] = []
    prev_assistant = _preceding_assistant_for_tool_result(messages, last_user)
    if prev_assistant is not None:
        recovered.append(prev_assistant)
    recovered.append(messages[last_user])
    omitted = len(messages) - len(recovered)
    return recovered, omitted, RECOVERY_NOTICE_CONTENT


def upstream_model_for(original_model: str, config: ProxyConfig) -> str:
    if original_model.startswith("deepseek-"):
        return original_model
    return config.upstream_model


def anthropic_cache_namespace(
    config: ProxyConfig,
    upstream_model: str,
    thinking: Any,
    reasoning_effort: Any,
    authorization: str | None = None,
) -> str:
    auth_hash = ""
    if authorization:
        auth_hash = hashlib.sha256(authorization.encode("utf-8")).hexdigest()
    payload = {
        "api_format": "anthropic",
        "base_url": config.upstream_base_url,
        "model": upstream_model,
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,
        "authorization_hash": auth_hash,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_anthropic_upstream_request(
    payload: dict[str, Any],
    config: ProxyConfig,
    store: ReasoningStore | None,
    authorization: str | None = None,
) -> PreparedAnthropicRequest:
    original_model = str(payload.get("model") or config.upstream_model)
    upstream_model = upstream_model_for(original_model, config)

    prepared: dict[str, Any] = {
        key: value
        for key, value in payload.items()
        if key in ANTHROPIC_SUPPORTED_REQUEST_FIELDS
    }
    prepared["model"] = upstream_model

    if config.thinking != "pass-through":
        prepared["thinking"] = {"type": config.thinking}

    thinking = prepared.get("thinking")
    thinking_enabled = (
        isinstance(thinking, dict) and thinking.get("type") == "enabled"
    )
    thinking_disabled = (
        isinstance(thinking, dict) and thinking.get("type") == "disabled"
    )

    if thinking_enabled:
        prepared["reasoning_effort"] = normalize_reasoning_effort(
            prepared.get("reasoning_effort") or config.reasoning_effort
        )

    cache_namespace = anthropic_cache_namespace(
        config,
        upstream_model,
        prepared.get("thinking"),
        prepared.get("reasoning_effort"),
        authorization,
    )

    messages, patched_count, missing_indexes = normalize_anthropic_messages(
        payload.get("messages"),
        store,
        cache_namespace,
        repair_thinking=thinking_enabled,
        keep_thinking=not thinking_disabled,
    )

    recovered_count = 0
    recovery_dropped = 0
    recovery_notice: str | None = None
    while missing_indexes and config.missing_reasoning_strategy == "recover":
        recovered_msgs, dropped, notice = recover_anthropic_messages(
            messages, missing_indexes
        )
        if not dropped:
            break
        recovered_count += len(missing_indexes)
        recovery_dropped += dropped
        if notice:
            recovery_notice = notice
        messages, patched_count, missing_indexes = (
            normalize_anthropic_messages(
                recovered_msgs,
                store,
                cache_namespace,
                repair_thinking=thinking_enabled,
                keep_thinking=not thinking_disabled,
            )
        )

    prepared["messages"] = messages

    if recovery_notice and "system" in prepared:
        existing_system = prepared["system"]
        if isinstance(existing_system, str):
            prepared["system"] = (
                f"{existing_system}\n\n{RECOVERY_SYSTEM_CONTENT}"
            )
        elif isinstance(existing_system, list):
            prepared["system"] = existing_system + [
                {"type": "text", "text": RECOVERY_SYSTEM_CONTENT}
            ]
    elif recovery_notice:
        prepared["system"] = RECOVERY_SYSTEM_CONTENT

    return PreparedAnthropicRequest(
        payload=prepared,
        original_model=original_model,
        upstream_model=upstream_model,
        cache_namespace=cache_namespace,
        patched_thinking_messages=patched_count,
        missing_thinking_messages=len(missing_indexes),
        recovered_thinking_messages=recovered_count,
        recovery_dropped_messages=recovery_dropped,
        recovery_notice=recovery_notice,
    )


def record_anthropic_response_reasoning(
    response_payload: dict[str, Any],
    store: ReasoningStore | None,
    request_messages: list[dict[str, Any]],
    cache_namespace: str = "",
) -> int:
    if store is None:
        return 0
    content = response_payload.get("content")
    if not isinstance(content, list):
        return 0
    thinking = extract_thinking_from_content(content)
    if thinking is None:
        return 0

    message = {
        "role": response_payload.get("role", "assistant"),
        "content": content,
    }
    scope = anthropic_conversation_scope(request_messages, cache_namespace)
    return store_anthropic_assistant_reasoning(store, message, scope)


def rewrite_anthropic_response_body(
    body: bytes,
    original_model: str,
    store: ReasoningStore | None,
    request_messages: list[dict[str, Any]],
    cache_namespace: str = "",
    content_prefix: str | None = None,
) -> bytes:
    response_payload = json.loads(body.decode("utf-8"))
    if isinstance(response_payload, dict):
        if content_prefix:
            prefix_anthropic_response_content(response_payload, content_prefix)
        record_anthropic_response_reasoning(
            response_payload, store, request_messages, cache_namespace
        )
        if "model" in response_payload:
            response_payload["model"] = original_model
    return json.dumps(
        response_payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def prefix_anthropic_response_content(
    response_payload: dict[str, Any], prefix: str
) -> bool:
    content = response_payload.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            block["text"] = prefix + (block.get("text") or "")
            return True
    content.insert(0, {"type": "text", "text": prefix})
    return True
