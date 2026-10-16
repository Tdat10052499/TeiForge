"""LLM providers used by the agent's tool-calling loop.

Messages use one provider-neutral format::

    {"role": "system" | "user" | "assistant" | "tool", "content": str}

Assistant messages may carry ``"tool_calls": [ToolCall, ...]``; tool messages carry
``"tool_call_id"`` and ``"name"``. Each provider converts this format to its own API.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from .config import AgentConfig
    from .tools import Tool

logger = logging.getLogger(__name__)

Message = Dict[str, Any]

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
PROVIDER_ENV_VAR = "TEIFORGE_PROVIDER"
MODEL_ENV_VAR = "TEIFORGE_MODEL"
BASE_URL_ENV_VAR = "TEIFORGE_BASE_URL"


class LLMError(RuntimeError):
    """Raised when a provider cannot be created or a request fails."""


@dataclass
class ToolCall:
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """A model reply: text, tool calls, or both."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can turn messages (and optional tools) into an :class:`LLMResponse`."""

    name: str
    model: str

    def complete(
        self, messages: List[Message], tools: Sequence[Tool] = ()
    ) -> LLMResponse: ...  # pragma: no cover


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        raise LLMError(f"Model returned tool arguments that are not valid JSON: {raw!r}") from None
    if not isinstance(parsed, dict):
        raise LLMError(f"Model returned tool arguments that are not an object: {raw!r}")
    return parsed


_TAGGED_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", flags=re.DOTALL)
_FENCED = re.compile(r"^```[A-Za-z]*\s*(.*?)\s*```$", flags=re.DOTALL)


def _json_values(text: str) -> Optional[List[Any]]:
    """Parse ``text`` as one JSON value, or as one JSON value per non-empty line."""
    try:
        return [json.loads(text)]
    except ValueError:
        pass
    values = []
    for line in text.splitlines():
        line = line.strip().rstrip(";")
        if not line:
            continue
        try:
            values.append(json.loads(line))
        except ValueError:
            return None
    return values or None


def _text_call(item: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("function"), dict):
        item = item["function"]
    name = item.get("name")
    arguments = item.get("arguments", item.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return name, arguments


def parse_text_tool_calls(content: Optional[str], tool_names: Sequence[str]) -> List[ToolCall]:
    """Recover tool calls that a model wrote as text instead of using the tool interface.

    Small local models sometimes answer with ``{"name": "file_reader", "parameters": {...}}``
    as plain text. The whole reply must be such a call (bare JSON, a fenced code block,
    ``<tool_call>`` tags or one call per line) and every call must name a tool in
    ``tool_names``; otherwise nothing is recovered, so a tool merely mentioned in prose is
    never run.
    """
    if not content or not tool_names:
        return []
    text = content.strip()
    if text.startswith("<|python_tag|>"):
        text = text[len("<|python_tag|>") :].strip()
    tagged = _TAGGED_CALL.findall(text)
    if tagged:
        if _TAGGED_CALL.sub("", text).strip():
            return []
        bodies = tagged
    else:
        fenced = _FENCED.match(text)
        bodies = [fenced.group(1) if fenced else text]

    calls: List[ToolCall] = []
    for body in bodies:
        values = _json_values(body)
        if values is None:
            return []
        for value in values:
            for item in value if isinstance(value, list) else [value]:
                parsed = _text_call(item)
                if parsed is None or parsed[0] not in tool_names:
                    return []
                calls.append(ToolCall(id=f"text_call_{len(calls) + 1}", name=parsed[0], arguments=parsed[1]))
    return calls


# -- OpenAI and compatible APIs ------------------------------------------------------


class OpenAIProvider:
    """OpenAI Chat Completions, or any compatible server through ``base_url``.

    ``base_url`` makes it work with Ollama (``http://localhost:11434/v1``), LM Studio,
    OpenRouter, vLLM and similar servers.
    """

    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        client: Any = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError(
                    "The OpenAI provider needs the 'openai' package: pip install 'teiforge[openai]'"
                ) from exc
            key = api_key or os.getenv("OPENAI_API_KEY") or ("not-needed" if base_url else None)
            if not key:
                raise LLMError("Set OPENAI_API_KEY to use the OpenAI provider.")
            client = OpenAI(api_key=key, base_url=base_url)
        self.client = client

    @staticmethod
    def is_reasoning_model(model: str) -> bool:
        """Return True for OpenAI reasoning models (o1, o3, o4-mini, gpt-5, ...)."""
        name = model.lower().rsplit("/", 1)[-1]
        return bool(re.match(r"^(o\d|gpt-5)", name))

    @staticmethod
    def convert_messages(messages: List[Message]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role == "assistant" and message.get("tool_calls"):
                converted.append(
                    {
                        "role": "assistant",
                        "content": message.get("content") or None,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments),
                                },
                            }
                            for call in message["tool_calls"]
                        ],
                    }
                )
            elif role == "tool":
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "content": message["content"],
                    }
                )
            else:
                converted.append({"role": role, "content": message["content"]})
        return converted

    def complete(self, messages: List[Message], tools: Sequence[Tool] = ()) -> LLMResponse:
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": self.convert_messages(messages),
        }
        if self.is_reasoning_model(self.model):
            # o-series and gpt-5 models reject max_tokens and custom temperatures.
            request["max_completion_tokens"] = self.max_tokens
        else:
            request["temperature"] = self.temperature
            request["max_tokens"] = self.max_tokens
        if tools:
            request["tools"] = [item.to_openai_schema() for item in tools]
        try:
            completion = self.client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001 - normalise SDK errors
            raise LLMError(f"OpenAI request failed: {type(exc).__name__}: {exc}") from exc

        message = completion.choices[0].message
        calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=_parse_arguments(call.function.arguments),
            )
            for call in (getattr(message, "tool_calls", None) or [])
        ]
        return LLMResponse(content=message.content, tool_calls=calls)


# -- Anthropic --------------------------------------------------------------------------


class AnthropicProvider:
    """Anthropic Messages API."""

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        client: Any = None,
    ) -> None:
        self.model = model
        self.temperature = min(temperature, 1.0)
        self.max_tokens = max_tokens
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise LLMError(
                    "The Anthropic provider needs the 'anthropic' package: pip install 'teiforge[anthropic]'"
                ) from exc
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError("Set ANTHROPIC_API_KEY to use the Anthropic provider.")
            client = Anthropic(api_key=key)
        self.client = client

    @staticmethod
    def convert_messages(messages: List[Message]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        system_parts: List[str] = []
        converted: List[Dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role == "system":
                system_parts.append(message["content"])
            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": message["content"],
                }
                if message.get("is_error"):
                    block["is_error"] = True
                previous = converted[-1] if converted else None
                if (
                    previous
                    and previous["role"] == "user"
                    and isinstance(previous["content"], list)
                    and all(item.get("type") == "tool_result" for item in previous["content"])
                ):
                    previous["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
            elif role == "assistant" and message.get("tool_calls"):
                content: List[Dict[str, Any]] = []
                if message.get("content"):
                    content.append({"type": "text", "text": message["content"]})
                content.extend(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                    for call in message["tool_calls"]
                )
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append({"role": role, "content": message["content"]})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, converted

    def complete(self, messages: List[Message], tools: Sequence[Tool] = ()) -> LLMResponse:
        system, converted = self.convert_messages(messages)
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [item.to_anthropic_schema() for item in tools]
        try:
            response = self.client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001 - normalise SDK errors
            raise LLMError(f"Anthropic request failed: {type(exc).__name__}: {exc}") from exc

        texts: List[str] = []
        calls: List[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=_parse_arguments(block.input)))
        return LLMResponse(content="".join(texts) or None, tool_calls=calls)


# -- discovery ----------------------------------------------------------------------------


def _package_available(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


def provider_from_config(config: AgentConfig) -> Optional[LLMProvider]:
    """Build the provider described by ``config`` and the environment.

    Returns ``None`` when no provider is configured. Raises :class:`LLMError` when a
    provider was requested explicitly but cannot be created.
    """
    choice = config.provider or os.getenv(PROVIDER_ENV_VAR) or "auto"
    choice = choice.strip().lower()
    base_url = config.base_url or os.getenv(BASE_URL_ENV_VAR)
    model_override = os.getenv(MODEL_ENV_VAR)
    model = model_override or config.model

    if choice == "none":
        return None
    if choice == "auto":
        if _package_available("openai") and (os.getenv("OPENAI_API_KEY") or base_url):
            choice = "openai"
        elif _package_available("anthropic") and os.getenv("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        else:
            return None

    if choice == "openai":
        return OpenAIProvider(
            model,
            base_url=base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    if choice == "anthropic":
        if not model_override and model.startswith(("gpt-", "o1", "o3", "o4")):
            model = DEFAULT_ANTHROPIC_MODEL
        return AnthropicProvider(model, temperature=config.temperature, max_tokens=config.max_tokens)
    raise LLMError(f"Unknown provider '{choice}'. Use openai, anthropic or none.")


__all__ = [
    "AnthropicProvider",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "OpenAIProvider",
    "ToolCall",
    "parse_text_tool_calls",
    "provider_from_config",
    "DEFAULT_ANTHROPIC_MODEL",
    "PROVIDER_ENV_VAR",
    "MODEL_ENV_VAR",
    "BASE_URL_ENV_VAR",
]
