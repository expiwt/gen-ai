"""
Фабрика OpenAI-совместимого клиента + лёгкий JSON-слой над .parse().

Все параметры берутся из окружения (или .env через python-dotenv).
Поддержан self-signed хост: verify=False, таймаут 200 с.

Ожидаемые переменные окружения:
 LLM_BASE_URL — базовый URL в духе https://host/v1
 LLM_AUTH_TOKEN — bearer-токен
 LLM_MODEL — имя модели для chat.completions

Fallback: если LLM_BASE_URL не задан — идём в публичный OpenAI и требуем OPENAI_API_KEY.

make_client() даёт drop-in API `client.chat.completions.create(
..., response_model=PydanticModel, max_retries=N)`.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any, Type, TypeVar, get_args, get_origin

import httpx
from openai import OpenAI
from pydantic import BaseModel, TypeAdapter

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

T = TypeVar("T")


def _make_openai_client() -> OpenAI:
    base = os.environ.get("LLM_BASE_URL")
    if base:
        key = os.environ.get("LLM_AUTH_TOKEN") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "LLM_AUTH_TOKEN не задан. Либо экспортируй токен, "
                "либо положи LLM_AUTH_TOKEN=*** в .env."
            )
        timeout = float(os.environ.get("LLM_TIMEOUT", "200"))
        http = httpx.Client(verify=False, timeout=timeout)
        return OpenAI(api_key=key, base_url=base, http_client=http)

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Ни LLM_BASE_URL, ни OPENAI_API_KEY не заданы. "
            "Сконфигурируй стенд через .env (см. .env.example)."
        )
    return OpenAI(api_key=key)


def get_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4.1-mini")


_HARMONY_RE = re.compile(r"<\|[^|>]*\|>")


def _thinking_off_payload() -> dict:
    if os.environ.get("LLM_THINKING", "off").lower() in ("on", "1", "true", "yes"):
        return {}
    return {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "reasoning_effort": "none",
    }


def _clean(text: str) -> str:
    text = _HARMONY_RE.sub("", text).strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_first_json(text: str):
    t = _clean(text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(t, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"В ответе не найдено валидного JSON: {text[:300]!r}")


class _Completions:
    def __init__(self, client: OpenAI):
        self._c = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        response_model: Type[T],
        max_retries: int = 1,
        temperature: float = 0.0,
        with_completion: bool = False,
        **kw: Any,
    ) -> T:
        wrap_list = get_origin(response_model) is list
        if wrap_list:
            item_type = get_args(response_model)[0]
            adapter = TypeAdapter(list[item_type])
            item_schema = TypeAdapter(item_type).json_schema()
            schema = {
                "type": "object",
                "properties": {"items": {"type": "array", "items": item_schema}},
                "required": ["items"],
            }
        else:
            adapter = TypeAdapter(response_model)
            schema = adapter.json_schema()

        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

        addendum = (
            f"\n\nОтвечай ОДНИМ валидным JSON-объектом по схеме:\n{schema_str}\n"
            "ТОЛЬКО JSON. Никакого текста до/после, никакого markdown, "
            "никаких комментариев, никаких повторных объектов."
        )
        if wrap_list:
            addendum += " Массив верни в поле `items`."

        msgs = [dict(m) for m in messages]
        sys_i = next((i for i, m in enumerate(msgs) if m["role"] == "system"), None)
        if sys_i is not None:
            msgs[sys_i]["content"] = msgs[sys_i]["content"] + addendum
        else:
            msgs.insert(0, {"role": "system", "content": addendum.lstrip()})

        thinking_kw = _thinking_off_payload()

        def _call(kw: dict):
            try:
                return self._c.chat.completions.create(
                    model=model,
                    messages=msgs,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    **kw,
                )
            except TypeError:
                safe = {k: v for k, v in kw.items() if k != "reasoning_effort"}
                return self._c.chat.completions.create(
                    model=model,
                    messages=msgs,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    **safe,
                )

        last_err: Exception | None = None
        raw: str = ""
        for _ in range(max_retries + 1):
            try:
                try:
                    resp = _call(thinking_kw)
                except Exception as sdk_err:
                    msg = str(sdk_err)
                    bad = (
                        "reasoning_effort" in msg
                        or "chat_template_kwargs" in msg
                        or "enable_thinking" in msg
                    )
                    if bad and thinking_kw:
                        thinking_kw = {}
                        resp = _call(thinking_kw)
                    else:
                        raise
                raw = resp.choices[0].message.content or ""
                obj = _extract_first_json(raw)
                if wrap_list and isinstance(obj, dict) and "items" in obj:
                    obj = obj["items"]
                result = adapter.validate_python(obj)
                if with_completion:
                    return result, resp
                return result
            except Exception as e:
                last_err = e
                msgs.append({"role": "assistant", "content": raw})
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            f"Невалидный ответ: {e}. "
                            "Верни ТОЛЬКО один корректный JSON по схеме."
                        ),
                    }
                )

        assert last_err is not None
        raise last_err


class _Chat:
    def __init__(self, client: OpenAI):
        self.completions = _Completions(client)


class JsonClient:
    """Drop-in клиент со structured outputs.

    API: client.chat.completions.create(
        response_model=PydanticModel, max_retries=N, ...
    )
    """

    def __init__(self, openai_client: OpenAI):
        self._c = openai_client
        self.chat = _Chat(openai_client)


def make_client() -> JsonClient:
    """Вернуть клиент с API `client.chat.completions.create(..., response_model=...)`."""
    return JsonClient(_make_openai_client())


class _RawCompletions:
    def __init__(self, inner):
        self._inner = inner

    def create(self, **kw: Any):
        thinking = _thinking_off_payload()

        def _call(extra: dict):
            try:
                return self._inner.create(**kw, **extra)
            except TypeError:
                safe = {k: v for k, v in extra.items() if k != "reasoning_effort"}
                return self._inner.create(**kw, **safe)

        try:
            return _call(thinking)
        except Exception as e:
            msg = str(e)
            bad = (
                "reasoning_effort" in msg
                or "chat_template_kwargs" in msg
                or "enable_thinking" in msg
            )
            if bad and thinking:
                return _call({})
            raise


class _RawChat:
    def __init__(self, inner):
        self.completions = _RawCompletions(inner.completions)


class RawClient:
    def __init__(self, openai_client: OpenAI):
        self._c = openai_client
        self.chat = _RawChat(openai_client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._c, name)


def make_raw_client() -> RawClient:
    return RawClient(_make_openai_client())
