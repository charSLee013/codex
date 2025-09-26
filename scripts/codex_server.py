#!/usr/bin/env python3
from __future__ import annotations
import json
import uuid
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

import fastapi
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import time

from codex_openai_common import (
    load_config,
    load_auth,
    pick_provider,
    endpoints,
    build_auth_headers,
    compute_instructions,
    tools_for_model,
)

VERIFIED_MODELS = [
    {
        "id": "gpt-5-codex",
        "capabilities": {
            "callable": True,
            "function_call": True,
            "reasoning": True,
            "effort": {"minimal": False, "low": True, "medium": True, "high": True},
        },
    },
    {
        "id": "gpt-5",
        "capabilities": {
            "callable": True,
            "function_call": True,
            "reasoning": False,
            "effort": {"minimal": False, "low": True, "medium": False, "high": True},
        },
    },
]

SUPPORTED_MODEL_IDS = [entry["id"] for entry in VERIFIED_MODELS]

async def _collect_openai_response_object(resp: httpx.Response, stream_ctx: Any) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Consume an upstream SSE response and return the final response payload."""
    final: Optional[Dict[str, Any]] = None
    reasoning_blocks: List[str] = []
    error_payload: Optional[Dict[str, Any]] = None
    try:
        async for line in resp.aiter_lines():
            if not line:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == "data: [DONE]":
                break
            if not line.startswith("data:"):
                continue
            payload_str = line[5:].strip()
            if not payload_str:
                continue
            try:
                event_obj = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            event_type = event_obj.get("type")
            if event_type == "response.error":
                err = event_obj.get("error")
                if isinstance(err, dict):
                    error_payload = err
                else:
                    error_payload = {"message": str(err or event_obj)}
                break
            if event_type in {"response.completed", "response.end"}:
                maybe_resp = event_obj.get("response")
                if isinstance(maybe_resp, dict):
                    final = maybe_resp
                    reasoning = maybe_resp.get("reasoning")
                    if isinstance(reasoning, dict):
                        encrypted = reasoning.get("encrypted_content")
                        if isinstance(encrypted, str) and encrypted:
                            reasoning_blocks.append(encrypted)
                if event_type == "response.completed":
                    break
    except httpx.StreamClosed:
        pass
    finally:
        await stream_ctx.__aexit__(None, None, None)
    if final is not None and reasoning_blocks:
        final["__aggregated_reasoning_content"] = reasoning_blocks
    return final, error_payload


def parse_effort_from_model(model: str) -> tuple[str, str | None]:
    parts = model.rsplit("-", 1)
    if len(parts) == 2 and parts[1].lower() in {"minimal", "low", "medium", "high"}:
        return parts[0], parts[1].lower()
    return model, None




def _anthropic_content_item_to_text(content_item: Any) -> str:
    if isinstance(content_item, str):
        return content_item
    if isinstance(content_item, dict):
        text = content_item.get("text")
        if isinstance(text, str):
            return text
    return ""


def _text_block_for_role(role: str, text: str) -> Dict[str, Any]:
    block_type = "input_text" if role in {"user", "system"} else "output_text"
    return {"type": block_type, "text": text}


def _anthropic_tool_use_block(item: Dict[str, Any]) -> Dict[str, Any]:
    tool_input = item.get("input")
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except json.JSONDecodeError:
            tool_input = {"raw": tool_input}
    if tool_input is None:
        tool_input = {}
    return {
        "type": "tool_use",
        "id": item.get("id") or str(uuid.uuid4()),
        "name": item.get("name"),
        "input": tool_input,
    }


def _anthropic_tool_result_block(item: Dict[str, Any]) -> Dict[str, Any]:
    tool_use_id = item.get("tool_use_id") or item.get("id") or str(uuid.uuid4())
    result: Dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": str(tool_use_id),
    }

    if "is_error" in item:
        result["is_error"] = bool(item.get("is_error"))

    content_items: List[Dict[str, Any]] = []
    payload = item.get("content")
    if isinstance(payload, list):
        for part in payload:
            text = _anthropic_content_item_to_text(part)
            if text:
                content_items.append({"type": "output_text", "text": text})
    else:
        text = _anthropic_content_item_to_text(payload)
        if text:
            content_items.append({"type": "output_text", "text": text})

    fallback_text = item.get("text")
    if not content_items and isinstance(fallback_text, str) and fallback_text:
        content_items.append({"type": "output_text", "text": fallback_text})

    if content_items:
        result["content"] = content_items

    return result


def _anthropic_messages_to_input(messages: Any, system_prompt: Optional[str]) -> List[dict]:
    result: List[dict] = []

    if system_prompt:
        result.append(
            {
                "type": "message",
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": str(system_prompt),
                    }
                ],
            }
        )

    if not isinstance(messages, list):
        return result

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant", "system"}:
            continue
        content = message.get("content")
        blocks: List[Dict[str, Any]] = []
        text_parts: List[str] = []

        def flush_text_buffer() -> None:
            if text_parts:
                text = "".join(text_parts)
                if text:
                    blocks.append(_text_block_for_role(role, text))
                text_parts.clear()

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type in {None, "text"}:
                        text = _anthropic_content_item_to_text(item)
                        if text:
                            text_parts.append(text)
                        continue
                    if item_type in {"input_text", "output_text"}:
                        flush_text_buffer()
                        text = item.get("text")
                        if isinstance(text, str) and text:
                            blocks.append({"type": item_type, "text": text})
                        continue
                    if item_type == "tool_use":
                        flush_text_buffer()
                        blocks.append(_anthropic_tool_use_block(item))
                        continue
                    if item_type == "tool_result":
                        flush_text_buffer()
                        blocks.append(_anthropic_tool_result_block(item))
                        continue
                    # Unknown item types are skipped.
                    continue
                text = _anthropic_content_item_to_text(item)
                if text:
                    text_parts.append(text)
        else:
            text = _anthropic_content_item_to_text(content)
            if text:
                text_parts.append(text)

        flush_text_buffer()

        if not blocks:
            continue

        result.append({"type": "message", "role": role, "content": blocks})

    return result


def _anthropic_output_content(output: List[dict]) -> List[dict]:
    content: List[dict] = []
    for block in output or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "message":
            for part in block.get("content", []) or []:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    content.append({"type": "text", "text": text})
        elif block_type == "tool_call":
            tool_call = block.get("tool_call") or {}
            arguments_raw = tool_call.get("arguments")
            try:
                arguments = json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw
            except json.JSONDecodeError:
                arguments = {"arguments": arguments_raw}
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id") or str(uuid.uuid4()),
                    "name": tool_call.get("name"),
                    "input": arguments,
                }
            )
        elif block_type == "function_call":
            # Some providers emit function_call items instead of tool_call
            fc = block.get("function_call") if isinstance(block.get("function_call"), dict) else {}
            # Fallback to top-level fields if nested object not present
            fid = fc.get("id") if fc else block.get("id")
            name = fc.get("name") if fc else block.get("name")
            arguments_raw = fc.get("arguments") if fc else block.get("arguments")
            try:
                arguments = json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw
            except json.JSONDecodeError:
                arguments = {"arguments": arguments_raw}
            if arguments is None:
                arguments = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": fid or str(uuid.uuid4()),
                    "name": name,
                    "input": arguments,
                }
            )
    return content


def _anthropic_reasoning_content(reasoning: Any) -> List[dict]:
    if not isinstance(reasoning, dict):
        return []
    encrypted = reasoning.get("encrypted_content")
    if isinstance(encrypted, str) and encrypted:
        return [{"type": "thinking", "text": encrypted}]
    return []


def _anthropic_tools_to_responses_tools(tools: Any) -> List[Dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = tool.get("input_schema")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description"),
                    "parameters": parameters,
                },
            }
        )
    return converted


def _convert_openai_stream_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(event, dict):
        return None
    event_type = event.get("type")
    index = event.get("output_index")
    if not isinstance(index, int):
        index = 0
    if event_type == "response.output_text.delta":
        delta_raw = event.get("delta")
        if isinstance(delta_raw, str):
            text = delta_raw
        elif isinstance(delta_raw, dict):
            text = delta_raw.get("text")
        else:
            text = None
        if isinstance(text, str) and text:
            return {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "text_delta",
                    "text": text,
                },
            }
    elif event_type == "response.output_text.done":
        # Signal the end of a text content block for this index
        return {"type": "content_block_stop", "index": index}
    elif event_type == "response.delta":
        delta_raw = event.get("delta")
        if isinstance(delta_raw, dict):
            thinking = delta_raw.get("thinking")
        else:
            thinking = None
        if isinstance(thinking, str) and thinking:
            return {
                "type": "thinking_delta",
                "index": index,
                "delta": {
                    "text": thinking,
                },
            }
    elif event_type == "response.tool_call.delta":
        # Streaming tool call arguments; map to a synthetic event we will expand
        # into Anthropic content_block_start/input_json_delta in the caller.
        tool_id = event.get("id") or (event.get("tool_call") or {}).get("id")
        name = event.get("name") or event.get("tool_name") or (event.get("tool_call") or {}).get("name")
        delta_raw = event.get("delta")
        args_chunk: Any
        if isinstance(delta_raw, str):
            # Some providers stream arguments as a raw string chunk
            args_chunk = delta_raw
        else:
            delta = delta_raw if isinstance(delta_raw, dict) else {}
            # The chunk may be under 'arguments', or another field; fallback to empty string
            args_chunk = delta.get("arguments")
            if args_chunk is None:
                args_chunk = delta.get("tool_arguments") or delta.get("arguments_delta")
        if isinstance(args_chunk, (dict, list)):
            try:
                import json as _json
                args_chunk = _json.dumps(args_chunk)
            except Exception:
                args_chunk = str(args_chunk)
        if not isinstance(args_chunk, str):
            args_chunk = ""
        return {
            "type": "tool_call_delta",
            "index": index,
            "id": tool_id,
            "name": name,
            "arguments": args_chunk,
        }
    elif event_type == "response.tool_call.done":
        # Signal end of current tool_use block on this index
        return {"type": "tool_call_stop", "index": index}
    elif event_type == "response.output_item.added":
        # Some providers announce function_call items via output_item.added
        item = event.get("item") or event.get("output_item") or {}
        if isinstance(item, dict):
            itype = item.get("type")
            if itype in {"function_call", "tool_call"}:
                tool_id = (
                    item.get("id")
                    or item.get("call_id")
                    or (item.get("function_call") or {}).get("id")
                )
                name = item.get("name") or (item.get("function_call") or {}).get("name") or item.get("tool_name")
                return {
                    "type": "tool_call_delta",
                    "index": index,
                    "id": tool_id,
                    "name": name,
                    "arguments": "",
                }
    elif event_type == "response.function_call_arguments.delta":
        # Streaming function_call arguments from Responses API
        delta_raw = event.get("delta")
        args_chunk: Any
        if isinstance(delta_raw, str):
            args_chunk = delta_raw
        elif isinstance(delta_raw, dict):
            # Prefer 'arguments' field; otherwise stringify the dict
            args_chunk = delta_raw.get("arguments")
            if args_chunk is None:
                try:
                    import json as _json
                    args_chunk = _json.dumps(delta_raw)
                except Exception:
                    args_chunk = str(delta_raw)
        else:
            args_chunk = ""
        if isinstance(args_chunk, (dict, list)):
            try:
                import json as _json
                args_chunk = _json.dumps(args_chunk)
            except Exception:
                args_chunk = str(args_chunk)
        if not isinstance(args_chunk, str):
            args_chunk = ""
        tool_id = event.get("id") or event.get("call_id") or (event.get("function_call") or {}).get("id")
        name = event.get("name") or (event.get("function_call") or {}).get("name") or event.get("tool_name")
        return {
            "type": "tool_call_delta",
            "index": index,
            "id": tool_id,
            "name": name,
            "arguments": args_chunk,
        }
    elif event_type == "response.function_call_arguments.done":
        return {"type": "tool_call_stop", "index": index}
    elif event_type == "response.output_item.done":
        # When a function_call item is marked done, close the tool block as a safeguard
        item = event.get("item") or event.get("output_item") or {}
        if isinstance(item, dict) and item.get("type") in {"function_call", "tool_call"}:
            return {"type": "tool_call_stop", "index": index}
    elif event_type in {"response.completed", "response.end"}:
        return {"type": "message_stop"}
    return None


def _anthropic_response_from_openai(data: Dict[str, Any], fallback_model: str) -> Dict[str, Any]:
    content = _anthropic_output_content(data.get("output") or [])
    content.extend(_anthropic_reasoning_content(data.get("reasoning")))
    return {
        "id": data.get("id", str(uuid.uuid4())),
        "type": "message",
        "role": "assistant",
        "model": data.get("model", fallback_model),
        "stop_reason": data.get("stop_reason", "end_turn"),
        "content": content or [{"type": "text", "text": ""}],
        "usage": data.get("usage"),
    }


def _chat_messages_to_responses_input(messages: Any) -> List[dict]:
    """Translate OpenAI Chat Completions messages into Responses input blocks.
    Supports roles: system, user, assistant, tool.
    - user/system text -> input_text
    - assistant text -> output_text; assistant tool_calls -> tool_use blocks
    - tool role -> mapped to a user message containing tool_result blocks
    """
    if not isinstance(messages, list):
        return []
    result: List[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        content = msg.get("content")
        blocks: List[Dict[str, Any]] = []

        def add_text(text: str, role_: str):
            if not isinstance(text, str) or not text:
                return
            blocks.append({"type": "input_text" if role_ in {"user", "system"} else "output_text", "text": text})

        # Handle assistant tool_calls
        if role == "assistant":
            # text content
            if isinstance(content, str):
                add_text(content, role)
            elif isinstance(content, list):
                for part in content:
                    t = part.get("text") if isinstance(part, dict) else (str(part) if isinstance(part, (str, int, float)) else None)
                    if isinstance(t, str):
                        add_text(t, role)
            # tool calls
            tool_calls = msg.get("tool_calls") or []
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    if tc.get("type") != "function":
                        continue
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    args_raw = fn.get("arguments")
                    args: Any = {}
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except json.JSONDecodeError:
                            args = {"raw": args_raw}
                    elif isinstance(args_raw, dict):
                        args = args_raw
                    blocks.append({"type": "tool_use", "id": tc.get("id") or str(uuid.uuid4()), "name": name, "input": args})
            if blocks:
                result.append({"type": "message", "role": "assistant", "content": blocks})
            continue

        # Map tool role to user tool_result
        if role == "tool":
            tc_id = msg.get("tool_call_id") or msg.get("id") or str(uuid.uuid4())
            text_payload = None
            if isinstance(content, str):
                text_payload = content
            elif isinstance(content, list):
                # concatenate any text items
                texts: List[str] = []
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        texts.append(part["text"])
                    elif isinstance(part, str):
                        texts.append(part)
                text_payload = "".join(texts) if texts else None
            tool_result: Dict[str, Any] = {"type": "tool_result", "tool_use_id": str(tc_id)}
            if isinstance(text_payload, str) and text_payload:
                tool_result["content"] = [{"type": "output_text", "text": text_payload}]
            result.append({"type": "message", "role": "user", "content": [tool_result]})
            continue

        # system/user messages
        if isinstance(content, str):
            add_text(content, role)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    add_text(part["text"], role)
                elif isinstance(part, str):
                    add_text(part, role)
        if blocks:
            result.append({"type": "message", "role": role, "content": blocks})
    return result


def _output_to_chat_message(output: List[dict]) -> Dict[str, Any]:
    """Collapse Responses output list into a Chat Completions message.
    Aggregates text and tool_calls into a single assistant message.
    """
    text_buf: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for item in output or []:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "message":
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    text_buf.append(part["text"])
        elif t == "tool_call":
            tc = item.get("tool_call") or {}
            fid = tc.get("id") or str(uuid.uuid4())
            name = tc.get("name")
            args_raw = tc.get("arguments")
            if isinstance(args_raw, (dict, list)):
                try:
                    args_raw = json.dumps(args_raw)
                except Exception:
                    args_raw = str(args_raw)
            if not isinstance(args_raw, str):
                args_raw = "{}"
            tool_calls.append({
                "id": fid,
                "type": "function",
                "function": {"name": name, "arguments": args_raw},
            })
        elif t == "function_call":
            # Convert function_call items to Chat tool_calls
            fc = item.get("function_call") if isinstance(item.get("function_call"), dict) else {}
            fid = (fc.get("id") if fc else item.get("id")) or str(uuid.uuid4())
            name = fc.get("name") if fc else item.get("name")
            args_raw = fc.get("arguments") if fc else item.get("arguments")
            if isinstance(args_raw, (dict, list)):
                try:
                    args_raw = json.dumps(args_raw)
                except Exception:
                    args_raw = str(args_raw)
            if not isinstance(args_raw, str):
                args_raw = "{}"
            tool_calls.append({
                "id": fid,
                "type": "function",
                "function": {"name": name, "arguments": args_raw},
            })
    return {
        "role": "assistant",
        "content": "".join(text_buf),
        "tool_calls": tool_calls or None,
    }

def make_payload(cfg: Dict[str, Any], model: str, user_body: Dict[str, Any], effort: str | None) -> Dict[str, Any]:
    base_model, _ = parse_effort_from_model(model)
    use_minimal_instructions = bool(user_body.pop("_use_minimal_instructions", False))
    tools = tools_for_model(cfg, base_model)
    if user_body.pop("_claude_strip_tools", False):
        tools = []

    # Build input from incoming body; accept either Responses-shaped items or a raw string prompt
    input_items = user_body.get("input")
    if not input_items:
        prompt_text = user_body.get("prompt") or user_body.get("query") or "ping"
        input_items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": str(prompt_text)}]}]

    if use_minimal_instructions:
        default_instructions = cfg.get("default_claude_instructions") or ""
    else:
        default_instructions = compute_instructions(
            base_model, include_apply_patch_tool=bool(cfg.get("include_apply_patch_tool", False))
        )
    provided_instructions = user_body.get("instructions") or cfg.get("default_instructions")
    if provided_instructions and default_instructions:
        instructions = f"{default_instructions}\n\n{provided_instructions}"
    elif provided_instructions:
        instructions = provided_instructions
    elif default_instructions:
        instructions = default_instructions
    else:
        instructions = "You are a helpful assistant."

    payload: Dict[str, Any] = {
        "model": base_model,
        "input": input_items,
        "instructions": instructions,
        "tools": tools,
        "tool_choice": user_body.get("tool_choice", "auto"),
        "parallel_tool_calls": bool(user_body.get("parallel_tool_calls", False)),
        "store": bool(user_body.get("store", False)),
        "stream": True if user_body.get("stream", True) else False,
    }

    include = user_body.get("include") or []

    # Reasoning: enable if effort or summary is provided, or when cfg enables summaries for this family.
    reasoning = user_body.get("reasoning") or {}
    if effort:
        reasoning["effort"] = effort
    if user_body.get("reasoning_summary"):
        reasoning["summary"] = str(user_body.get("reasoning_summary")).lower()
    if reasoning:
        payload["reasoning"] = reasoning
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")

    if include:
        payload["include"] = include

    # text controls can be passed through from user_body if present
    if user_body.get("text"):
        payload["text"] = user_body["text"]

    # If caller supplied tools, prefer them over defaults and ensure Responses format
    if isinstance(user_body.get("tools"), list):
        user_tools = user_body.get("tools")

        def _looks_like_responses_tools(tools_list: List[Any]) -> bool:
            for t in tools_list:
                if not isinstance(t, dict):
                    return False
                if t.get("type") != "function" or not isinstance(t.get("function"), dict):
                    return False
            return len(tools_list) > 0

        if _looks_like_responses_tools(user_tools):
            payload["tools"] = user_tools  # already in correct schema
        else:
            converted = _anthropic_tools_to_responses_tools(user_tools)
            if converted:
                payload["tools"] = converted
    elif tools:
        payload["tools"] = tools

    return payload


app = fastapi.FastAPI(title="Codex Server", version="0.1.0")

CFG = load_config()
PROVIDER = pick_provider(CFG)
EPS = endpoints(PROVIDER)
AUTH = load_auth()
BASE_HEADERS = build_auth_headers(PROVIDER, AUTH)
CLIENT: Optional[httpx.AsyncClient] = None

# Determine debug mode before configuring handlers so the logger level is correct.
DEBUG_CLAUDE = bool(os.getenv("CODEX_DEBUG_CLAUDE"))

LOGGER = logging.getLogger("codex_server")
if not LOGGER.handlers:
    # Logger level gates all handlers; use DEBUG when CODEX_DEBUG_CLAUDE=1
    LOGGER.setLevel(logging.DEBUG if DEBUG_CLAUDE else logging.INFO)
    fmt = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Console handler for info-level summaries
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)

    # File handlers (default to workdir if not specified)
    base_dir = os.getenv("CODEX_LOG_DIR") or os.getcwd()

    err_path = os.getenv("CODEX_ERROR_LOG") or os.path.join(base_dir, "codex_error.log")
    eh = RotatingFileHandler(err_path, maxBytes=5*1024*1024, backupCount=2, encoding="utf-8", delay=True)
    eh.setLevel(logging.WARNING)  # only errors/warnings
    eh.setFormatter(fmt)
    LOGGER.addHandler(eh)

    if DEBUG_CLAUDE:
        dbg_path = os.getenv("CODEX_DEBUG_LOG") or os.path.join(base_dir, "codex_claude_debug.log")
        dh = RotatingFileHandler(dbg_path, maxBytes=10*1024*1024, backupCount=2, encoding="utf-8", delay=True)
        dh.setLevel(logging.DEBUG)
        dh.setFormatter(fmt)
        # Filter out WARNING and above to keep debug file strictly for DEBUG/INFO
        class _MaxLevelFilter(logging.Filter):
            def __init__(self, max_level: int) -> None:
                super().__init__(name="")
                self.max_level = max_level
            def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
                return record.levelno <= self.max_level
        dh.addFilter(_MaxLevelFilter(logging.INFO))
        LOGGER.addHandler(dh)

def _cdbg(msg: str, *args: Any) -> None:
    if DEBUG_CLAUDE:
        LOGGER.debug(msg, *args)

def _probe(kind: str, payload: Dict[str, Any]) -> None:
    # Minimal, always-on probe for Claude endpoint; writes only metadata (no user text)
    try:
        path = os.path.join(os.getcwd(), "codex_claude_probe.log")
        with open(path, "a", encoding="utf-8") as f:
            safe = {k: payload.get(k) for k in ("model", "stream", "tool_choice")}
            tools = []
            for t in (payload.get("tools") or []):
                n = None
                if isinstance(t, dict):
                    n = t.get("name") or (t.get("function") or {}).get("name")
                if n:
                    tools.append(str(n))
            safe["tools"] = tools
            f.write(json.dumps({"kind": kind, **safe}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _log_error_minimal(path: str, status: int, model: Optional[str], tools: Any, tool_choice: Any, note: str = "") -> None:
    """Write a single-line WARNING with only metadata for troubleshooting.
    - No user text or payload bodies.
    """
    try:
        tool_names: List[str] = []
        if isinstance(tools, list):
            for t in tools:
                if isinstance(t, dict):
                    n = t.get("name") or (t.get("function") or {}).get("name")
                    if isinstance(n, str) and n:
                        tool_names.append(n)
        meta = {
            "path": path,
            "status": status,
            "model": model,
            "tool_choice": tool_choice if isinstance(tool_choice, (str, dict)) else None,
            "tools": tool_names,
            "note": note or None,
        }
        LOGGER.warning("error: %s", json.dumps(meta, ensure_ascii=False))
    except Exception:
        # Never raise from logging
        pass


@app.on_event("startup")
async def _startup():
    global CLIENT
    CLIENT = httpx.AsyncClient(http2=True, timeout=300.0)


@app.on_event("shutdown")
async def _shutdown():
    global CLIENT
    if CLIENT:
        await CLIENT.aclose()
        CLIENT = None


@app.get("/v1/models")
async def list_models():
    # Do not hit upstream /models – return only the models we have verified.
    return JSONResponse({"data": VERIFIED_MODELS})


@app.post("/v1/responses")
async def post_responses(req: Request):
    assert CLIENT is not None
    body = await req.json()
    model = body.get("model") or CFG.get("model", "gpt-5-codex")
    base_model, effort = parse_effort_from_model(model)
    payload = make_payload(CFG, base_model, body, effort)

    conv_id = str(uuid.uuid4())
    headers = {
        **BASE_HEADERS,
        "OpenAI-Beta": "responses=experimental",
        "conversation_id": conv_id,
        "session_id": conv_id,
        "Content-Type": "application/json",
    }
    payload["prompt_cache_key"] = conv_id

    client_stream = bool(payload.get("stream", True))
    payload["stream"] = True
    headers["Accept"] = "text/event-stream"

    stream_ctx = CLIENT.stream("POST", EPS["responses"], headers=headers, json=payload)
    resp = await stream_ctx.__aenter__()
    close_resp = True
    try:
        if resp.status_code >= 400:
            try:
                err = await resp.json()
            except Exception:
                raw_err = await resp.aread()
                text_err = raw_err.decode("utf-8", "ignore")
                try:
                    err = json.loads(text_err)
                except Exception:
                    err = {"error": {"message": text_err}}
            _log_error_minimal(
                path="/v1/responses",
                status=resp.status_code,
                model=base_model,
                tools=payload.get("tools"),
                tool_choice=payload.get("tool_choice"),
                note="upstream_error",
            )
            return JSONResponse(err, status_code=resp.status_code)

        if client_stream:
            close_resp = False

            async def event_iter():
                try:
                    async for line in resp.aiter_lines():
                        if line:
                            yield (line + "\n").encode("utf-8")
                        else:
                            yield b"\n"
                except httpx.StreamClosed:
                    pass
                finally:
                    await stream_ctx.__aexit__(None, None, None)

            return StreamingResponse(event_iter(), media_type="text/event-stream")

        response_obj, error_payload = await _collect_openai_response_object(resp, stream_ctx)
        close_resp = False
        if response_obj is not None:
            if "__aggregated_reasoning_content" in response_obj:
                response_obj.setdefault("reasoning", {})["encrypted_content"] = response_obj.pop("__aggregated_reasoning_content")
            return JSONResponse(response_obj)
        if error_payload is not None:
            return JSONResponse({"error": error_payload}, status_code=502)
        return JSONResponse({"error": {"message": "Upstream stream ended without completion"}}, status_code=502)
    except httpx.RequestError as e:
        _log_error_minimal(
            path="/v1/responses",
            status=502,
            model=base_model,
            tools=payload.get("tools"),
            tool_choice=payload.get("tool_choice"),
            note=f"request_error:{e}",
        )
        return JSONResponse({"error": {"message": str(e), "type": "request_error"}}, status_code=502)
    finally:
        if close_resp:
            await stream_ctx.__aexit__(None, None, None)


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    """Compatibility endpoint that adapts Chat Completions requests to Responses.
    - Translates messages/tools to Responses schema
    - Streams back OpenAI Chat Completions chunks, or returns a final Chat payload
    """
    assert CLIENT is not None
    body = await req.json()
    model = body.get("model") or CFG.get("model", "gpt-5-codex")
    base_model, effort = parse_effort_from_model(model)
    client_stream = bool(body.get("stream", False))

    # Convert Chat messages to Responses input
    # Map Chat-style tool_choice to Responses format
    tool_choice_in = body.get("tool_choice")
    mapped_tool_choice_c = None
    if isinstance(tool_choice_in, dict) and tool_choice_in.get("type") == "function":
        fn = tool_choice_in.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn.get("name"):
            mapped_tool_choice_c = {"type": "function", "name": fn.get("name")}
    elif isinstance(tool_choice_in, str) and tool_choice_in in ("auto", "none"):
        mapped_tool_choice_c = tool_choice_in

    user_body: Dict[str, Any] = {
        "input": _chat_messages_to_responses_input(body.get("messages")),
        "stream": client_stream,
        "tool_choice": mapped_tool_choice_c if mapped_tool_choice_c is not None else body.get("tool_choice"),
        "parallel_tool_calls": bool(body.get("parallel_tool_calls", False)),
        "store": bool(body.get("store", False)),
    }

    # Tools: prefer `tools`; fallback to legacy `functions`
    tools = body.get("tools")
    if not tools and isinstance(body.get("functions"), list):
        tools = []
        for fn in body.get("functions"):
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            tools.append({"type": "function", "function": {"name": fn.get("name"), "description": fn.get("description"), "parameters": fn.get("parameters") or {"type": "object", "properties": {}}}})
    if tools:
        # Ensure Responses format and backfill top-level fields for providers that validate tools[0].name
        normalized: List[Dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict):
                fn = t["function"]
                name = t.get("name") or fn.get("name")
                params = t.get("parameters") or fn.get("parameters")
                desc = t.get("description") or fn.get("description")
                if name:
                    t.setdefault("name", name)
                if params is not None:
                    t.setdefault("parameters", params)
                if desc is not None:
                    t.setdefault("description", desc)
                normalized.append(t)
            else:
                # Accept Anthropic-style {name,input_schema}
                name = t.get("name") if isinstance(t, dict) else None
                schema = t.get("input_schema") if isinstance(t, dict) else None
                if isinstance(name, str) and name:
                    normalized.append({
                        "type": "function",
                        "name": name,
                        "parameters": schema if isinstance(schema, dict) else {"type":"object","properties":{}},
                        "function": {
                            "name": name,
                            "parameters": schema if isinstance(schema, dict) else {"type":"object","properties":{}},
                        },
                    })
        if normalized:
            user_body["tools"] = normalized

    payload = make_payload(CFG, base_model, user_body, effort)
    payload["stream"] = True

    conv_id = str(uuid.uuid4())
    headers = {
        **BASE_HEADERS,
        "OpenAI-Beta": "responses=experimental",
        "conversation_id": conv_id,
        "session_id": conv_id,
        "Content-Type": "application/json",
    }
    headers["Accept"] = "text/event-stream"
    payload["prompt_cache_key"] = conv_id

    stream_ctx = CLIENT.stream("POST", EPS["responses"], headers=headers, json=payload)
    resp = await stream_ctx.__aenter__()
    close_resp = True
    try:
        if resp.status_code >= 400:
            try:
                err = await resp.json()
            except Exception:
                raw_err = await resp.aread()
                text_err = raw_err.decode("utf-8", "ignore")
                try:
                    err = json.loads(text_err)
                except Exception:
                    err = {"error": {"message": text_err}}
            return JSONResponse(err, status_code=resp.status_code)

        if client_stream:
            close_resp = False

            async def event_iter():
                created = int(time.time())
                sent_role = False
                tool_id_to_idx: Dict[str, int] = {}
                next_tool_idx = 0
                try:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        stripped = line.strip()
                        if stripped == "data: [DONE]":
                            # Final stop chunk
                            chunk = {
                                "id": conv_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": base_model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            }
                            yield ("data: " + json.dumps(chunk) + "\n\n").encode("utf-8")
                            yield b"data: [DONE]\n\n"
                            return
                        if not line.startswith("data:"):
                            continue
                        payload_str = line[5:].strip()
                        if not payload_str:
                            continue
                        try:
                            event_obj = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue
                        mapped = _convert_openai_stream_event(event_obj)
                        if mapped is None:
                            continue
                        t = mapped.get("type")

                        # Emit initial role chunk once
                        if not sent_role and t in {"content_block_delta", "tool_call_delta"}:
                            role_chunk = {
                                "id": conv_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": base_model,
                                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                            }
                            yield ("data: " + json.dumps(role_chunk) + "\n\n").encode("utf-8")
                            sent_role = True

                        if t == "content_block_delta":
                            delta = mapped.get("delta") or {}
                            text = delta.get("text") if delta.get("type") == "text_delta" else None
                            if isinstance(text, str) and text:
                                chunk = {
                                    "id": conv_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": base_model,
                                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                                }
                                yield ("data: " + json.dumps(chunk) + "\n\n").encode("utf-8")
                            continue

                        if t == "tool_call_delta":
                            tool_id = mapped.get("id") or str(uuid.uuid4())
                            name = mapped.get("name")
                            args = mapped.get("arguments", "")
                            if tool_id not in tool_id_to_idx:
                                tool_id_to_idx[tool_id] = next_tool_idx
                                next_tool_idx += 1
                            idx = tool_id_to_idx[tool_id]
                            delta_obj: Dict[str, Any] = {
                                "tool_calls": [
                                    {
                                        "index": idx,
                                        "id": tool_id,
                                        "type": "function",
                                        "function": {"arguments": args},
                                    }
                                ]
                            }
                            if name and idx == tool_id_to_idx[tool_id]:
                                delta_obj["tool_calls"][0]["function"]["name"] = name
                            chunk = {
                                "id": conv_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": base_model,
                                "choices": [{"index": 0, "delta": delta_obj, "finish_reason": None}],
                            }
                            yield ("data: " + json.dumps(chunk) + "\n\n").encode("utf-8")
                            continue

                        if t == "message_stop":
                            chunk = {
                                "id": conv_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": base_model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            }
                            yield ("data: " + json.dumps(chunk) + "\n\n").encode("utf-8")
                            yield b"data: [DONE]\n\n"
                            return
                except httpx.StreamClosed:
                    pass
                finally:
                    await stream_ctx.__aexit__(None, None, None)

            return StreamingResponse(event_iter(), media_type="text/event-stream")

        response_obj, error_payload = await _collect_openai_response_object(resp, stream_ctx)
        close_resp = False
        if response_obj is None:
            if error_payload is not None:
                _log_error_minimal(
                    path="/v1/chat/completions",
                    status=502,
                    model=base_model,
                    tools=user_body.get("tools"),
                    tool_choice=user_body.get("tool_choice"),
                    note="stream_error_payload",
                )
                return JSONResponse({"error": error_payload}, status_code=502)
            _log_error_minimal(
                path="/v1/chat/completions",
                status=502,
                model=base_model,
                tools=user_body.get("tools"),
                tool_choice=user_body.get("tool_choice"),
                note="upstream_stream_ended",
            )
            return JSONResponse({"error": {"message": "Upstream stream ended without completion"}}, status_code=502)

        message = _output_to_chat_message(response_obj.get("output") or [])
        reasoning_blocks = response_obj.pop("__aggregated_reasoning_content", None)
        if reasoning_blocks:
            message["reasoning_content"] = reasoning_blocks
        created = int(time.time())
        chat: Dict[str, Any] = {
            "id": response_obj.get("id", "chatcmpl-" + conv_id),
            "object": "chat.completion",
            "created": response_obj.get("created_at", created),
            "model": response_obj.get("model", base_model),
            "choices": [
                {
                    "index": 0,
                    "message": {k: v for k, v in message.items() if v is not None},
                    "finish_reason": "stop",
                }
            ],
        }
        if isinstance(response_obj.get("usage"), dict):
            chat["usage"] = response_obj["usage"]
        return JSONResponse(chat)
    except httpx.RequestError as e:
        _log_error_minimal(
            path="/v1/chat/completions",
            status=502,
            model=base_model,
            tools=user_body.get("tools"),
            tool_choice=user_body.get("tool_choice"),
            note=f"request_error:{e}",
        )
        return JSONResponse({"error": {"message": str(e), "type": "request_error"}}, status_code=502)
    finally:
        if close_resp:
            await stream_ctx.__aexit__(None, None, None)

@app.post("/claude/v1/messages")
async def claude_messages(req: Request):
    assert CLIENT is not None
    body = await req.json()
    requested_model = body.get("model")
    if requested_model:
        base_for_check, _ = parse_effort_from_model(requested_model)
        if base_for_check not in SUPPORTED_MODEL_IDS:
            _log_error_minimal(
                path="/claude/v1/messages",
                status=400,
                model=requested_model,
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                note="unsupported_model_param",
            )
            return JSONResponse(
                {
                    "error": {
                        "message": "Unsupported model",
                        "supported_models": SUPPORTED_MODEL_IDS,
                    }
                },
                status_code=400,
            )
        model = requested_model
    else:
        model = CFG.get("model", SUPPORTED_MODEL_IDS[0])

    base_model, effort = parse_effort_from_model(model)
    if base_model not in SUPPORTED_MODEL_IDS:
        _log_error_minimal(
            path="/claude/v1/messages",
            status=400,
            model=model,
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
            note="unsupported_model_default",
        )
        return JSONResponse(
            {
                "error": {
                    "message": "Unsupported model",
                    "supported_models": SUPPORTED_MODEL_IDS,
                }
            },
            status_code=400,
        )
    stream_requested = bool(body.get("stream", False))

    raw_messages = body.get("messages") or []
    system_chunks: List[str] = []
    filtered_messages: List[Any] = []

    for msg in raw_messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    text = _anthropic_content_item_to_text(part)
                    if text:
                        system_chunks.append(text)
            else:
                text = _anthropic_content_item_to_text(content)
                if text:
                    system_chunks.append(text)
            continue
        filtered_messages.append(msg)

    system_prompt = body.get("system")
    if isinstance(system_prompt, str) and system_prompt:
        system_chunks.insert(0, system_prompt)

    # Move system prompt content into the latest user message's text
    if system_chunks:
        sys_text = "\n\n".join(system_chunks).strip()
        last_user_idx = -1
        for i in range(len(filtered_messages) - 1, -1, -1):
            if isinstance(filtered_messages[i], dict) and filtered_messages[i].get("role") == "user":
                last_user_idx = i
                break
        original_text = ""
        if last_user_idx >= 0:
            c = filtered_messages[last_user_idx].get("content")
            parts: List[str] = []
            if isinstance(c, list):
                for it in c:
                    t = _anthropic_content_item_to_text(it)
                    if t:
                        parts.append(t)
            else:
                t = _anthropic_content_item_to_text(c)
                if t:
                    parts.append(t)
            original_text = "\n".join(parts).strip()
            new_text = (
                "请你遵循下面的规则以及工具调用\n"
                f"{sys_text}\n\n---\n\n"
                "下面是我的原始请求\n"
                f"{original_text}"
            )
            filtered_messages[last_user_idx]["content"] = new_text
        else:
            new_text = (
                "请你遵循下面的规则以及工具调用\n"
                f"{sys_text}\n\n---\n\n"
                "下面是我的原始请求\n"
            )
            filtered_messages.append({"role": "user", "content": new_text})

    # Map Anthropic tool_choice to Responses format if specified
    tool_choice_in = body.get("tool_choice")
    mapped_tool_choice = None
    if isinstance(tool_choice_in, dict) and tool_choice_in.get("type") == "tool":
        n = tool_choice_in.get("name")
        if isinstance(n, str) and n:
            mapped_tool_choice = {"type": "function", "name": n}
    elif isinstance(tool_choice_in, str):
        if tool_choice_in in ("auto", "none"):
            mapped_tool_choice = tool_choice_in

    user_body: Dict[str, Any] = {
        "input": _anthropic_messages_to_input(filtered_messages, None),
        "stream": stream_requested,
        "tool_choice": mapped_tool_choice if mapped_tool_choice is not None else body.get("tool_choice"),
        "parallel_tool_calls": bool(
            body.get("parallel_tool_calls") or body.get("allow_parallel_tool_use", False)
        ),
        "store": bool(body.get("store", False)),
    }

    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        user_body["reasoning"] = reasoning

    payload = make_payload(CFG, base_model, user_body, effort)
    tools = _anthropic_tools_to_responses_tools(body.get("tools"))
    if tools:
        # Add top-level name/parameters for providers that validate tools[0].name
        for t in tools:
            fn = t.get("function") or {}
            name = t.get("name") or fn.get("name")
            if name:
                t.setdefault("name", name)
            params = fn.get("parameters")
            if params is not None:
                t.setdefault("parameters", params)
            desc = fn.get("description")
            if desc is not None:
                t.setdefault("description", desc)
        payload["tools"] = tools
    # Debug summary (no request body or user content)
    _tool_names = []
    for t in (payload.get("tools") or []):
        n = t.get("name") or (t.get("function") or {}).get("name")
        if n:
            _tool_names.append(str(n))
    _cdbg("Claude payload summary: model=%s stream=%s tools=%s tool_choice=%s", base_model, True, _tool_names, payload.get("tool_choice"))
    _probe("claude_payload", {"model": base_model, "stream": True, "tool_choice": payload.get("tool_choice"), "tools": payload.get("tools")})
    payload["stream"] = True

    conv_id = str(uuid.uuid4())
    headers = {
        **BASE_HEADERS,
        "OpenAI-Beta": "responses=experimental",
        "conversation_id": conv_id,
        "session_id": conv_id,
        "Content-Type": "application/json",
    }
    headers["Accept"] = "text/event-stream"
    payload["prompt_cache_key"] = conv_id

    stream_ctx = CLIENT.stream("POST", EPS["responses"], headers=headers, json=payload)
    resp = await stream_ctx.__aenter__()
    close_resp = True
    try:
        if resp.status_code >= 400:
            _log_error_minimal(
                path="/claude/v1/messages",
                status=resp.status_code,
                model=base_model,
                tools=payload.get("tools"),
                tool_choice=payload.get("tool_choice"),
                note="upstream_error",
            )
            try:
                err = await resp.json()
            except Exception:
                raw_err = await resp.aread()
                text_err = raw_err.decode("utf-8", "ignore")
                try:
                    err = json.loads(text_err)
                except Exception:
                    err = {"error": {"message": text_err}}
            return JSONResponse(err, status_code=resp.status_code)

        if stream_requested:
            close_resp = False

            async def event_iter():
                terminated = False
                message_started = False
                # Track currently open content block type per index: 'text' or 'tool_use'
                open_block: Dict[int, str] = {}
                tool_meta: Dict[int, Dict[str, Any]] = {}
                ev_counts_up: Dict[str, int] = {}
                ev_counts_mapped: Dict[str, int] = {}

                def emit(obj: Dict[str, Any]):
                    return ("data: " + json.dumps(obj) + "\n\n").encode("utf-8")

                def maybe_message_start() -> Optional[bytes]:
                    nonlocal message_started
                    if message_started:
                        return None
                    message_started = True
                    # Minimal message object to satisfy Anthropic SSE contract
                    msg = {
                        "type": "message_start",
                        "message": {
                            "id": str(uuid.uuid4()),
                            "type": "message",
                            "role": "assistant",
                            "model": base_model,
                            "content": [],
                        },
                    }
                    return emit(msg)

                def open_text(index: int) -> Optional[bytes]:
                    # If a different block is open, close it first
                    if open_block.get(index) and open_block[index] != "text":
                        stop = {"type": "content_block_stop", "index": index}
                        open_block.pop(index, None)
                        # yield stop then start
                        return emit(stop) + emit({
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {"type": "text", "text": ""},
                        })
                    if open_block.get(index) == "text":
                        return None
                    open_block[index] = "text"
                    return emit({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    })

                def open_tool(index: int, tool_id: Optional[str], name: Optional[str]) -> bytes:
                    if not tool_id:
                        tool_id = str(uuid.uuid4())
                    if not name:
                        name = "tool"
                    # Close different block if needed
                    parts: list[bytes] = []
                    if open_block.get(index) and open_block[index] != "tool_use":
                        parts.append(emit({"type": "content_block_stop", "index": index}))
                    if open_block.get(index) != "tool_use":
                        open_block[index] = "tool_use"
                        tool_meta[index] = {"id": tool_id, "name": name}
                        parts.append(
                            emit(
                                {
                                    "type": "content_block_start",
                                    "index": index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tool_id,
                                        "name": name,
                                        "input": {},
                                    },
                                }
                            )
                        )
                    return b"".join(parts)

                def close_index(index: int) -> Optional[bytes]:
                    if open_block.get(index):
                        open_block.pop(index, None)
                        tool_meta.pop(index, None)
                        return emit({"type": "content_block_stop", "index": index})
                    return None

                try:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        stripped = line.strip()
                        if stripped == "data: [DONE]":
                            if not terminated:
                                # Close any open blocks then stop
                                for idx in list(open_block.keys()):
                                    buf = close_index(idx)
                                    if buf:
                                        yield buf
                                yield emit({"type": "message_stop"})
                                terminated = True
                            continue
                        if not line.startswith("data:"):
                            continue
                        payload_str = line[5:].strip()
                        if not payload_str:
                            continue
                        try:
                            event_obj = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue
                        et = event_obj.get("type")
                        if isinstance(et, str):
                            ev_counts_up[et] = ev_counts_up.get(et, 0) + 1
                        mapped = _convert_openai_stream_event(event_obj)
                        if mapped is None:
                            continue

                        t = mapped.get("type")
                        if isinstance(t, str):
                            ev_counts_mapped[t] = ev_counts_mapped.get(t, 0) + 1
                        # Ensure message_start precedes any deltas/starts
                        if t in {"content_block_delta", "thinking_delta", "tool_call_delta"}:
                            first = maybe_message_start()
                            if first:
                                yield first

                        if t == "content_block_delta":
                            idx = int(mapped.get("index", 0))
                            start = open_text(idx)
                            if start:
                                yield start
                            yield emit(mapped)
                            continue

                        if t == "content_block_stop":
                            idx = int(mapped.get("index", 0))
                            buf = close_index(idx)
                            if buf:
                                # We already emit our own stop to keep state consistent
                                yield buf
                            else:
                                # If nothing open, still forward the stop event
                                yield emit(mapped)
                            continue

                        if t == "tool_call_delta":
                            idx = int(mapped.get("index", 0))
                            pre = open_tool(idx, mapped.get("id"), mapped.get("name"))
                            if pre:
                                yield pre
                            # Emit input_json_delta carrying the partial arguments JSON
                            yield emit(
                                {
                                    "type": "content_block_delta",
                                    "index": idx,
                                    "delta": {"type": "input_json_delta", "partial_json": mapped.get("arguments", "")},
                                }
                            )
                            continue

                        if t == "tool_call_stop":
                            idx = int(mapped.get("index", 0))
                            buf = close_index(idx)
                            if buf:
                                yield buf
                            else:
                                yield emit({"type": "content_block_stop", "index": idx})
                            continue

                        if t == "thinking_delta":
                            yield emit(mapped)
                            continue

                        if t == "message_stop":
                            if not terminated:
                                # Close open blocks first
                                for idx in list(open_block.keys()):
                                    buf = close_index(idx)
                                    if buf:
                                        yield buf
                                yield emit(mapped)
                                terminated = True
                            continue

                        # Fallback: forward as-is
                        yield emit(mapped)
                except httpx.StreamClosed:
                    pass
                finally:
                    await stream_ctx.__aexit__(None, None, None)
                    _cdbg("Claude upstream events: %s; mapped events: %s", ev_counts_up, ev_counts_mapped)
                    _probe("claude_stream_counts", {"model": base_model, "stream": True, "tool_choice": payload.get("tool_choice"), "tools": payload.get("tools"), "up": ev_counts_up, "mapped": ev_counts_mapped})

                if not terminated:
                    # Close any open blocks then stop
                    for idx in list(open_block.keys()):
                        buf = close_index(idx)
                        if buf:
                            yield buf
                    yield emit({"type": "message_stop"})

            return StreamingResponse(event_iter(), media_type="text/event-stream")

        response_obj, error_payload = await _collect_openai_response_object(resp, stream_ctx)
        close_resp = False
        if response_obj is not None:
            reasoning_blocks = response_obj.pop("__aggregated_reasoning_content", None)
            converted = _anthropic_response_from_openai(response_obj, model)
            if reasoning_blocks:
                converted.setdefault("reasoning", {}).setdefault("encrypted_content", reasoning_blocks)
            # Debug summary for non-stream
            out = response_obj.get("output") or []
            if isinstance(out, list):
                kinds = {}
                for it in out:
                    if isinstance(it, dict):
                        k = it.get("type")
                        if isinstance(k, str):
                            kinds[k] = kinds.get(k, 0) + 1
                _cdbg("Claude aggregated output kinds: %s", kinds)
                _probe("claude_nonstream_kinds", {"model": base_model, "stream": False, "tool_choice": payload.get("tool_choice"), "tools": payload.get("tools"), "kinds": kinds})
            return JSONResponse(converted)
        if error_payload is not None:
            _log_error_minimal(
                path="/claude/v1/messages",
                status=502,
                model=base_model,
                tools=payload.get("tools"),
                tool_choice=payload.get("tool_choice"),
                note="stream_error_payload",
            )
            return JSONResponse({"error": error_payload}, status_code=502)
        _log_error_minimal(
            path="/claude/v1/messages",
            status=502,
            model=base_model,
            tools=payload.get("tools"),
            tool_choice=payload.get("tool_choice"),
            note="upstream_stream_ended",
        )
        return JSONResponse({"error": {"message": "Upstream stream ended without completion"}}, status_code=502)
    except httpx.RequestError as e:
        _log_error_minimal(
            path="/claude/v1/messages",
            status=502,
            model=base_model if 'base_model' in locals() else None,
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
            note=f"request_error:{e}",
        )
        return JSONResponse({"error": {"message": str(e), "type": "request_error"}}, status_code=502)
    finally:
        if close_resp:
            await stream_ctx.__aexit__(None, None, None)


@app.post("/claude/v1/messages/count_tokens")
async def claude_count_tokens(req: Request):
    body = await req.json()
    requested_model = body.get("model")
    if requested_model:
        base_for_check, _ = parse_effort_from_model(requested_model)
        if base_for_check not in SUPPORTED_MODEL_IDS:
            _log_error_minimal(
                path="/claude/v1/messages/count_tokens",
                status=400,
                model=requested_model,
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                note="unsupported_model_param",
            )
            return JSONResponse(
                {
                    "error": {
                        "message": "Unsupported model",
                        "supported_models": SUPPORTED_MODEL_IDS,
                    }
                },
                status_code=400,
            )
        model = requested_model
    else:
        model = CFG.get("model", SUPPORTED_MODEL_IDS[0])

    base_model, effort = parse_effort_from_model(model)
    if base_model not in SUPPORTED_MODEL_IDS:
        _log_error_minimal(
            path="/claude/v1/messages/count_tokens",
            status=400,
            model=model,
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
            note="unsupported_model_default",
        )
        return JSONResponse(
            {
                "error": {
                    "message": "Unsupported model",
                    "supported_models": SUPPORTED_MODEL_IDS,
                }
            },
            status_code=400,
        )

    raw_messages = body.get("messages") or []
    system_chunks: List[str] = []
    filtered_messages: List[Any] = []
    for msg in raw_messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    text = _anthropic_content_item_to_text(part)
                    if text:
                        system_chunks.append(text)
            else:
                text = _anthropic_content_item_to_text(content)
                if text:
                    system_chunks.append(text)
            continue
        filtered_messages.append(msg)

    system_prompt = body.get("system")
    if isinstance(system_prompt, str) and system_prompt:
        system_chunks.insert(0, system_prompt)

    if system_chunks:
        sys_text = "\n\n".join(system_chunks).strip()
        last_user_idx = -1
        for i in range(len(filtered_messages) - 1, -1, -1):
            if isinstance(filtered_messages[i], dict) and filtered_messages[i].get("role") == "user":
                last_user_idx = i
                break
        original_text = ""
        if last_user_idx >= 0:
            c = filtered_messages[last_user_idx].get("content")
            parts: List[str] = []
            if isinstance(c, list):
                for it in c:
                    t = _anthropic_content_item_to_text(it)
                    if t:
                        parts.append(t)
            else:
                t = _anthropic_content_item_to_text(c)
                if t:
                    parts.append(t)
            original_text = "\n".join(parts).strip()
            new_text = (
                "请你遵循下面的规则以及工具调用\n"
                f"{sys_text}\n\n---\n\n"
                "下面是我的原始请求\n"
                f"{original_text}"
            )
            filtered_messages[last_user_idx]["content"] = new_text
        else:
            new_text = (
                "请你遵循下面的规则以及工具调用\n"
                f"{sys_text}\n\n---\n\n"
                "下面是我的原始请求\n"
            )
            filtered_messages.append({"role": "user", "content": new_text})

    # Map tool_choice for count_tokens path as well
    tool_choice_in2 = body.get("tool_choice")
    mapped_tool_choice2 = None
    if isinstance(tool_choice_in2, dict) and tool_choice_in2.get("type") == "tool":
        n = tool_choice_in2.get("name")
        if isinstance(n, str) and n:
            mapped_tool_choice2 = {"type": "function", "name": n}
    elif isinstance(tool_choice_in2, str):
        if tool_choice_in2 in ("auto", "none"):
            mapped_tool_choice2 = tool_choice_in2

    user_body: Dict[str, Any] = {
        "input": _anthropic_messages_to_input(filtered_messages, None),
        "stream": False,
        "tool_choice": mapped_tool_choice2 if mapped_tool_choice2 is not None else body.get("tool_choice"),
        "parallel_tool_calls": bool(body.get("parallel_tool_calls") or body.get("allow_parallel_tool_use", False)),
        "store": bool(body.get("store", False)),
    }
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        user_body["reasoning"] = reasoning

    payload = make_payload(CFG, base_model, user_body, effort)
    tools = _anthropic_tools_to_responses_tools(body.get("tools"))
    if tools:
        payload["tools"] = tools

    estimate = max(1, len(json.dumps(payload)) // 4)
    return JSONResponse({"input_tokens": estimate})



# You can also run this file directly for local testing. It binds to 127.0.0.1:45443.
if __name__ == "__main__":
    try:
        import uvicorn  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "uvicorn is required to run codex_server directly.\n"
            "Install with: pip install uvicorn fastapi \"httpx[http2]\"\n"
            f"Import error: {e}"
        )

    uvicorn.run(app, host="127.0.0.1", port=45443, log_level="info")
