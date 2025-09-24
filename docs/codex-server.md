# Codex Server (Responses Proxy)

A minimal OpenAI-compatible server that exposes `/v1/models` and `/v1/responses`.
It mirrors Codex CLI’s request shape and lets you encode reasoning effort in the
model name (e.g., `gpt-5-high`, `gpt-5-codex-low`).

## Run

Install runtime deps (once): `pip install fastapi uvicorn httpx`

Start server (ASGI):

```
uvicorn scripts.codex-server:app --host 0.0.0.0 --port 8000 --workers 1
```

Requires `~/.codex/config.toml` and `~/.codex/auth.json` as used by Codex CLI.

## Endpoints

- `GET /v1/models`
  - Returns a list of model ids. Falls back to a conservative list if the
    upstream `/models` is unavailable.

- `POST /v1/responses`
  - Accepts an OpenAI Responses request body. If the `model` is suffixed with
    a reasoning effort (e.g., `-minimal|-low|-medium|-high`), the server will
    set `reasoning.effort` accordingly and forward to the configured provider.
  - Streams events back using `text/event-stream`.

### Examples

- List models (concurrent GET):

  `seq 1 5 | xargs -n1 -P5 -I{} curl -sS http://127.0.0.1:8000/v1/models | head -n1`

- Streamed Responses (effort from model name):

  `curl -N -H 'Content-Type: application/json' \\
    --data '{"model":"gpt-5-codex-low","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"ping"}]}]}' \\
    http://127.0.0.1:8000/v1/responses`

- Non-stream JSON:

  `curl -sS -H 'Content-Type: application/json' \\
    --data '{"model":"gpt-5-high","stream":false,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"json please"}]}]}' \\
    http://127.0.0.1:8000/v1/responses | jq .`

## Effort-in-model mapping

- `gpt-5-high` → `{ model: "gpt-5", reasoning: { effort: "high" } }`
- `gpt-5-codex-low` → `{ model: "gpt-5-codex", reasoning: { effort: "low" } }`

## Implementation notes

- Shared logic for auth/config/instructions/tools lives in `scripts/codex_openai_common.py`.
- Server code: `scripts/codex-server.py` (FastAPI + httpx async streaming passthrough).
- The server preserves Codex’s instruction/tool assembly so downstream behaviour matches the CLI.

## Concurrency

- FastAPI runs on an event loop; each inbound request is handled asynchronously.
- Upstream streaming uses a shared `httpx.AsyncClient` with HTTP/2 enabled and connection pooling.
- Increase `--workers` for multi-process concurrency when CPU-bound work appears (SSE proxy本身是 I/O 密集型，通常 1 个 worker 足够起步)。

## Reasoning (enable_think)

- 默认不“自动开启” reasoning；只有以下情况才会启用：
  - 模型名带 effort 后缀（如 `gpt-5-high`、`gpt-5-codex-low`）→ 服务端将 effort 写入 `reasoning.effort`
  - 请求体显式提供 `reasoning` 字段（例如 `{ "reasoning": { "effort": "low", "summary": "detailed" } }`）
- 若你的上游 provider/模型对白名单 effort 有限制（例如仅允许 `medium`），上游会返回 4xx；直接按返回体调整努力值即可。

## Troubleshooting

- 400 Unsupported model：上游白名单不接受该 `model` id，换成对方支持的 slug。
- 400 with verbosity/effort：上游不接受该值（例如只允许 `medium`），调整或移除相关字段。
- 4xx/5xx 错误体：服务直接透传上游 JSON，便于排障。
