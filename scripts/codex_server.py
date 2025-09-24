#!/usr/bin/env python3
from __future__ import annotations
import json
import uuid
from typing import Any, Dict, Optional

import fastapi
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from codex_openai_common import (
    load_config,
    load_auth,
    pick_provider,
    endpoints,
    build_auth_headers,
    compute_instructions,
    tools_for_model,
    DEFAULT_FALLBACK_MODELS,
)



def parse_effort_from_model(model: str) -> tuple[str, str | None]:
    parts = model.rsplit("-", 1)
    if len(parts) == 2 and parts[1].lower() in {"minimal", "low", "medium", "high"}:
        return parts[0], parts[1].lower()
    return model, None


def make_payload(cfg: Dict[str, Any], model: str, user_body: Dict[str, Any], effort: str | None) -> Dict[str, Any]:
    base_model, _ = parse_effort_from_model(model)
    tools = tools_for_model(cfg, base_model)
    instructions = compute_instructions(base_model, include_apply_patch_tool=bool(cfg.get("include_apply_patch_tool", False)))

    # Build input from incoming body; accept either Responses-shaped items or a raw string prompt
    input_items = user_body.get("input")
    if not input_items:
        prompt_text = user_body.get("prompt") or user_body.get("query") or "ping"
        input_items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": str(prompt_text)}]}]

    payload: Dict[str, Any] = {
        "model": base_model,
        "instructions": instructions,
        "input": input_items,
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

    return payload


app = fastapi.FastAPI(title="Codex Server", version="0.1.0")

CFG = load_config()
PROVIDER = pick_provider(CFG)
EPS = endpoints(PROVIDER)
AUTH = load_auth()
BASE_HEADERS = build_auth_headers(PROVIDER, AUTH)
CLIENT: Optional[httpx.AsyncClient] = None


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
    assert CLIENT is not None
    url = EPS["models"]
    try:
        r = await CLIENT.get(url, headers=BASE_HEADERS)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            obj = r.json()
            data = obj.get("data") if isinstance(obj, dict) else None
            slugs = [m.get("id") for m in (data or []) if isinstance(m, dict) and m.get("id")]
            slugs = [s for s in slugs if isinstance(s, str)]
            if slugs:
                return JSONResponse({"data": [{"id": m} for m in slugs]})
    except Exception:
        pass
    # Fallback list
    return JSONResponse({"data": [{"id": m} for m in DEFAULT_FALLBACK_MODELS]})


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
        "Accept": "text/event-stream" if payload.get("stream", True) else "application/json",
        "conversation_id": conv_id,
        "session_id": conv_id,
        "Content-Type": "application/json",
    }
    payload["prompt_cache_key"] = conv_id

    try:
        async with CLIENT.stream("POST", EPS["responses"], headers=headers, json=payload) as resp:
            if resp.status_code >= 400:
                # bubble provider error
                try:
                    err = await resp.json()
                except Exception:
                    err = {"error": {"message": await resp.aread()}}
                return JSONResponse(err, status_code=resp.status_code)

            if payload.get("stream", True):
                async def event_iter():
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        # Upstream already provides proper SSE lines; forward as-is
                        yield (line + "\n").encode("utf-8")

                return StreamingResponse(event_iter(), media_type="text/event-stream")

            # non-stream JSON
            try:
                data = await resp.json()
                return JSONResponse(data)
            except Exception:
                raw = await resp.aread()
                return fastapi.Response(content=raw, media_type=resp.headers.get("content-type", "application/json"))
    except httpx.RequestError as e:
        return JSONResponse({"error": {"message": str(e), "type": "request_error"}}, status_code=502)


# Note: This module is meant to be run via uvicorn: `uvicorn scripts.codex_server:app ...`
# Intentionally no CLI entrypoints beyond the ASGI `app` object to keep surface minimal.
