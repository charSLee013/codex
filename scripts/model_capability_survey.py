#!/usr/bin/env python3
from __future__ import annotations

"""
Unified capability survey for Codex-compatible providers.

Goals
- Discover available models (via /models with fallback list)
- For each model, determine:
  * callable: whether a basic Responses API request succeeds
  * function_call: whether a function_call output item appears
  * function_tool: the tool name used when function_call occurs
  * deep_thinking: whether reasoning events are emitted
- Emit a Markdown table and save it to --out (default: reports/model_capabilities.md)

This merges and supersedes the roles of:
- scripts/probe_models_function_call.py
- scripts/survey_models.py
- scripts/test_openai_responses.py (subset relevant to building payloads)
- scripts/codex_request.py (subset relevant to tool declarations)

Notes
- Intentionally uses the Responses API regardless of config's wire_api so that
  function_call and reasoning can be detected consistently.
- Declares both a generic function tool ("shell") and the special "local_shell"
  for broader provider compatibility. No tool_outputs are sent back; we only
  detect whether the model produces a function_call.
"""

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

# Conservative/default fallback set used if /models is unavailable
from codex_openai_common import (
    load_config,
    load_auth,
    pick_provider,
    endpoints,
    build_auth_headers,
    compute_instructions,
    tools_for_model,
    get_models as get_models_common,
    DEFAULT_FALLBACK_MODELS,
)


# All config/auth/provider helpers are imported from codex_openai_common.


# ---------- Model family + instructions ----------



def get_models(
    session: requests.Session,
    base_headers: Dict[str, str],
    eps: Dict[str, str],
    limit: Optional[int],
    explicit: Optional[List[str]] = None,
) -> List[str]:
    if explicit:
        return explicit
    return get_models_common(session, base_headers, eps, limit, DEFAULT_FALLBACK_MODELS)



def make_payload(cfg: Dict[str, Any], model: str, include_reasoning: bool, echo_cmd: list[str]) -> Dict[str, Any]:
    tools = tools_for_model(cfg, model)
    instructions = compute_instructions(model, include_apply_patch_tool=bool(cfg.get("include_apply_patch_tool", False)))
    prompt = (
        "If you support function calling, immediately call a tool to run the command "
        f"{json.dumps(echo_cmd)} with no extra commentary."
    )
    payload: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": False,
        "stream": True,
    }
    if include_reasoning:
        effort = (cfg.get("model_reasoning_effort") or "medium").lower()
        summary = (cfg.get("model_reasoning_summary") or "auto").lower()
        payload["reasoning"] = {"summary": summary, "effort": effort}
        payload["include"] = ["reasoning.encrypted_content"]
    verbosity = cfg.get("model_verbosity")
    if verbosity:
        payload["text"] = {"verbosity": str(verbosity).lower()}
    return payload


def probe(
    session: requests.Session,
    eps: Dict[str, str],
    base_headers: Dict[str, str],
    cfg: Dict[str, Any],
    model: str,
    include_reasoning: bool,
    timeout: int,
    echo_cmd: list[str],
) -> Dict[str, Any]:
    result = {
        "model": model,
        "http": None,
        "callable": False,
        "function_call": False,
        "function_tool": None,
        "deep_thinking": False,
        "notes": "",
    }
    payload = make_payload(cfg, model, include_reasoning, echo_cmd)
    headers = {
        **base_headers,
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "OpenAI-Beta": "responses=experimental",
    }
    # Match CLI: attach conversation/session headers and prompt_cache_key
    conv_id = str(__import__("uuid").uuid4())
    headers = {**headers, "conversation_id": conv_id, "session_id": conv_id}
    payload["prompt_cache_key"] = conv_id
    try:
        resp = session.post(eps["responses"], headers=headers, json=payload, stream=True, timeout=timeout)
    except requests.RequestException as e:
        result["notes"] = f"request_error: {type(e).__name__}"
        return result
    result["http"] = resp.status_code
    if resp.status_code >= 400:
        # Try to include server's error detail to diagnose shape mismatches
        err_txt = ""
        try:
            err_txt = resp.text
        except Exception:
            pass
        # Auto‑adjust: some providers only accept text.verbosity = "medium" for GPT‑5
        if "Unsupported value: 'high' is not supported" in (err_txt or "") or "verbosity" in (err_txt or ""):
            # downgrade or remove text controls and retry once
            text = payload.get("text") or {}
            text["verbosity"] = "medium"
            payload["text"] = text
            # new conversation/session for retry
            conv_id2 = str(__import__("uuid").uuid4())
            headers["conversation_id"] = conv_id2
            headers["session_id"] = conv_id2
            payload["prompt_cache_key"] = conv_id2
            try:
                resp2 = session.post(eps["responses"], headers=headers, json=payload, stream=True, timeout=timeout)
                result["http"] = resp2.status_code
                if resp2.status_code < 400:
                    result["callable"] = True
                    for raw in resp2.iter_lines():
                        if not raw:
                            continue
                        line = raw.decode("utf-8", errors="replace")
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            ev = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        t = ev.get("type")
                        if t and t.startswith("response.reasoning"):
                            result["deep_thinking"] = True
                        if t == "response.output_item.added":
                            item = ev.get("item", {})
                            if item.get("type") == "function_call":
                                result["function_call"] = True
                                result["function_tool"] = item.get("name") or item.get("tool_name")
                    return result
                else:
                    try:
                        err2 = resp2.text
                    except Exception:
                        err2 = ""
                    result["notes"] = ("http_error: " + (err2[:180] if err2 else ""))
                    return result
            except requests.RequestException as e2:
                result["notes"] = f"request_error_after_adjust: {type(e2).__name__}"
                return result
        result["notes"] = ("http_error: " + (err_txt[:180] if err_txt else ""))
        return result
    result["callable"] = True
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t and t.startswith("response.reasoning"):
            result["deep_thinking"] = True
        if t == "response.output_item.added":
            item = ev.get("item", {})
            if item.get("type") == "function_call":
                result["function_call"] = True
                result["function_tool"] = item.get("name") or item.get("tool_name")
    return result


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| 模型 | 可调用 | 支持 Function Call | 工具名 | 支持深度思考 | minimal | low | medium | high | HTTP | 备注 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    def yn(v: bool) -> str:
        return "✅" if v else "❌"
    for r in rows:
        es = r.get("effort_support") or {}
        lines.append(
            f"| {r['model']} | {yn(r['callable'])} | {yn(r['function_call'])} | "
            f"{(r['function_tool'] or '-')} | {yn(r['deep_thinking'])} | "
            f"{yn(es.get('minimal', False))} | {yn(es.get('low', False))} | {yn(es.get('medium', False))} | {yn(es.get('high', False))} | "
            f"{r['http']} | {r['notes']} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Discover models and survey callable/function-call/reasoning support.")
    ap.add_argument("models", nargs="*", help="Specific model ids to test; omit to auto-discover or use fallback list")
    ap.add_argument("--limit", type=int, default=12, help="Limit when discovering from /models")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("reports/model_capabilities.md"))
    ap.add_argument("--no-reasoning", action="store_true", help="Do not include reasoning fields in the payload")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout (seconds) for each model probe")
    ap.add_argument("--echo", nargs="+", default=["echo", "hello-fc"], help="Command array passed to the tool call request")
    ap.add_argument("--efforts", nargs="+", default=["minimal","low","medium","high"], help="Reasoning efforts to probe when reasoning is enabled")
    args = ap.parse_args()

    # Pre-flight: rely on common loaders to validate presence
    cfg = load_config()
    provider = pick_provider(cfg)
    eps = endpoints(provider)
    auth = load_auth()
    session = requests.Session()
    base_headers = build_auth_headers(provider, auth)

    # Discover models
    models = get_models(session, base_headers, eps, args.limit, args.models or None)
    print(f"Probing provider '{provider['__id']}' at {eps['responses']}")
    print(f"Models: {', '.join(models)}")

    # Probe each model
    results: List[Dict[str, Any]] = []
    for m in models:
        r = probe(session, eps, base_headers, cfg, m, include_reasoning=not args.no_reasoning, timeout=args.timeout, echo_cmd=args.echo)
        print(
            f"- {m}: callable={r['callable']} function_call={r['function_call']} "
            f"deep={r['deep_thinking']} http={r['http']} tool={(r['function_tool'] or '-')}"
        )
        effort_support: Dict[str, bool] = {}
        if not args.no_reasoning and r.get("http", 0) < 400:
            for eff in args.efforts:
                cfg2 = dict(cfg)
                cfg2["model_reasoning_effort"] = eff
                rr = probe(session, eps, base_headers, cfg2, m, include_reasoning=True, timeout=args.timeout, echo_cmd=args.echo)
                effort_support[eff] = bool(rr.get("callable") and rr.get("deep_thinking"))
                print(f"  effort={eff}: http={rr['http']} deep={rr['deep_thinking']}")
                time.sleep(0.2)
        r["effort_support"] = effort_support
        results.append(r)
        time.sleep(0.25)

    # Emit Markdown
    md = to_markdown(results)
    out_path: pathlib.Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"\nMarkdown saved to: {out_path}")
    print(md)


if __name__ == "__main__":
    main()
    main()
