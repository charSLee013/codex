#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional

# Determine Codex home as read-only config directory.
# Honor CODEX_HOME environment variable to allow development/testing without
# touching the user's real ~/.codex directory.
HOME = pathlib.Path.home()
_ENV_CODEX_HOME = os.getenv("CODEX_HOME")
CODEX_HOME = pathlib.Path(_ENV_CODEX_HOME).expanduser() if _ENV_CODEX_HOME else (HOME / ".codex")
CONFIG_PATH = CODEX_HOME / "config.toml"
AUTH_PATH = CODEX_HOME / "auth.json"

try:
    import tomllib as toml_reader  # py311+
except ModuleNotFoundError:  # Python < 3.11
    import tomli as toml_reader  # type: ignore


def read_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8")


def load_auth() -> Dict[str, Any]:
    return json.loads(read_text(AUTH_PATH))


def load_config() -> Dict[str, Any]:
    return toml_reader.loads(read_text(CONFIG_PATH))


def pick_provider(cfg: Dict[str, Any]) -> Dict[str, Any]:
    providers = cfg.get("model_providers") or {}
    provider_id = cfg.get("model_provider", "openai")
    provider = providers.get(provider_id)
    if not provider:
        raise SystemExit(f"Provider '{provider_id}' not defined in config.toml")
    return {**provider, "__id": provider_id}


def endpoints(provider: Dict[str, Any]) -> Dict[str, str]:
    base = str(provider.get("base_url", "https://api.openai.com/v1")).rstrip("/")
    return {
        "responses": f"{base}/responses",
        "models": f"{base}/models",
    }

# Conservative/default fallback set used if /models is unavailable
DEFAULT_FALLBACK_MODELS = [
    "o3",
    "o4-mini",
    "codex-mini-latest",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-05-13",
    "gpt-3.5-turbo",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "gpt-5",
    "gpt-5-codex",
]


def build_auth_headers(provider: Dict[str, Any], auth: Dict[str, Any]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    # Treat OpenAI-style auth as required by default. This avoids surprising
    # differences when the config omits `requires_openai_auth`.
    if provider.get("requires_openai_auth", True):
        tokens = auth.get("tokens") or {}
        access = tokens.get("access_token")
        if access:
            headers["Authorization"] = f"Bearer {access}"
            acc_id = tokens.get("chatgpt_account_id") or auth.get("chatgpt_account_id") or auth.get("account_id")
            if acc_id:
                headers["chatgpt-account-id"] = str(acc_id)
        elif auth.get("OPENAI_API_KEY"):
            headers["Authorization"] = f"Bearer {auth['OPENAI_API_KEY']}"
        else:
            raise SystemExit("requires_openai_auth=true but no usable token or OPENAI_API_KEY present")
    else:
        key = auth.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("auth.json missing OPENAI_API_KEY")
        headers["Authorization"] = f"Bearer {key}"
    return headers


# ---------- Instructions + tools (mirrors codex-rs) ----------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "codex-rs" / "core"
PROMPT_DEFAULT = CORE_DIR / "prompt.md"
PROMPT_GPT5_CODEX = CORE_DIR / "gpt_5_codex_prompt.md"
APPLY_PATCH_TOOL_INSTRUCTIONS = (
    (REPO_ROOT / "codex-rs" / "apply-patch" / "apply_patch_tool_instructions.md").read_text(encoding="utf-8")
)
LARK_GRAMMAR = (CORE_DIR / "src" / "tool_apply_patch.lark").read_text(encoding="utf-8")


class ModelFamily:
    def __init__(self, slug: str):
        self.slug = slug
        self.family = slug
        self.supports_reasoning_summaries = False
        self.needs_special_apply_patch_instructions = False
        self.uses_local_shell_tool = False
        self.apply_patch_tool_type: Optional[str] = None
        self.base_instructions_path = PROMPT_DEFAULT

        if slug.startswith("o3"):
            self.family = "o3"
            self.supports_reasoning_summaries = True
            self.needs_special_apply_patch_instructions = True
        elif slug.startswith("o4-mini"):
            self.family = "o4-mini"
            self.supports_reasoning_summaries = True
            self.needs_special_apply_patch_instructions = True
        elif slug.startswith("codex-mini-latest"):
            self.family = "codex-mini-latest"
            self.supports_reasoning_summaries = True
            self.uses_local_shell_tool = True
            self.needs_special_apply_patch_instructions = True
        elif slug.startswith("gpt-4.1"):
            self.family = "gpt-4.1"
            self.needs_special_apply_patch_instructions = True
        elif slug.startswith("gpt-oss") or slug.startswith("openai/gpt-oss"):
            self.family = "gpt-oss"
            self.apply_patch_tool_type = "function"
        elif slug.startswith("gpt-4o"):
            self.family = "gpt-4o"
            self.needs_special_apply_patch_instructions = True
        elif slug.startswith("gpt-3.5"):
            self.family = "gpt-3.5"
            self.needs_special_apply_patch_instructions = True
        elif slug.startswith("codex-") or slug.startswith("gpt-5-codex"):
            self.family = slug
            self.supports_reasoning_summaries = True
            self.base_instructions_path = PROMPT_GPT5_CODEX
        elif slug.startswith("gpt-5"):
            self.family = "gpt-5"
            self.supports_reasoning_summaries = True
            self.needs_special_apply_patch_instructions = True

    def base_instructions(self) -> str:
        return self.base_instructions_path.read_text(encoding="utf-8")


def need_apply_patch_append(mf: ModelFamily, include_apply_patch_tool: bool) -> bool:
    return mf.needs_special_apply_patch_instructions and not include_apply_patch_tool


def compute_instructions(model: str, include_apply_patch_tool: bool) -> str:
    mf = ModelFamily(model)
    base = mf.base_instructions()
    if need_apply_patch_append(mf, include_apply_patch_tool):
        return f"{base}\n{APPLY_PATCH_TOOL_INSTRUCTIONS}"
    return base


def plan_tool() -> dict:
    return {
        "type": "function",
        "name": "update_plan",
        "description": "Updates the task plan. Provide an optional explanation and a list of plan items, each with a step and status. At most one step can be in_progress at a time.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {"type": "string"},
                "plan": {
                    "type": "array",
                    "description": "The list of steps",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string"},
                            "status": {"type": "string", "description": "One of: pending, in_progress, completed"},
                        },
                        "required": ["step", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["plan"],
            "additionalProperties": False,
        },
    }


def shell_tool() -> dict:
    return {
        "type": "function",
        "name": "shell",
        "description": "Runs a shell command and returns its output.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "array", "items": {"type": "string"}, "description": "The command to execute"},
                "workdir": {"type": "string", "description": "The working directory to execute the command in"},
                "timeout_ms": {"type": "number", "description": "The timeout for the command in milliseconds"},
                "with_escalated_permissions": {"type": "boolean", "description": "Whether to request escalated permissions. Set to true if command needs to be run without sandbox restrictions"},
                "justification": {"type": "string", "description": "Only set if with_escalated_permissions is true. 1-sentence explanation of why we want to run this command."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    }


def view_image_tool() -> dict:
    return {
        "type": "function",
        "name": "view_image",
        "description": "Attach a local image (by filesystem path) to the conversation context for this turn.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Local filesystem path to an image file"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    }


def apply_patch_freeform_tool() -> dict:
    return {
        "type": "custom",
        "name": "apply_patch",
        "description": "Use the `apply_patch` tool to edit files",
        "format": {"type": "grammar", "syntax": "lark", "definition": LARK_GRAMMAR},
    }


def apply_patch_json_tool() -> dict:
    return {
        "type": "function",
        "name": "apply_patch",
        "description": "Use the `apply_patch` tool to edit files.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {"input": {"type": "string", "description": "The entire contents of the apply_patch command"}},
            "required": ["input"],
            "additionalProperties": False,
        },
    }


def tools_for_model(cfg: Dict[str, Any], model: str) -> List[dict]:
    mf = ModelFamily(model)
    include_plan = True
    include_apply_patch = bool(cfg.get("include_apply_patch_tool", False))
    include_view_image = bool(cfg.get("include_view_image_tool", True))

    tools: List[dict] = []
    tools.append({"type": "local_shell"} if mf.uses_local_shell_tool else shell_tool())
    if include_plan:
        tools.append(plan_tool())
    if mf.apply_patch_tool_type == "function":
        tools.append(apply_patch_json_tool())
    elif include_apply_patch:
        tools.append(apply_patch_freeform_tool())
    if include_view_image:
        tools.append(view_image_tool())
    return tools


def get_models(session, base_headers: Dict[str, str], eps: Dict[str, str], limit: Optional[int], fallback: List[str]) -> List[str]:
    url = eps["models"]
    try:
        r = session.get(url, headers=base_headers, timeout=20)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            obj = r.json()
            data = obj.get("data") if isinstance(obj, dict) else None
            slugs = [m.get("id") for m in (data or []) if isinstance(m, dict) and m.get("id")]
            slugs = [s for s in slugs if isinstance(s, str)]
            if limit:
                slugs = slugs[: limit]
            if slugs:
                return slugs
    except Exception:
        pass
    return fallback[: limit] if limit else list(fallback)
