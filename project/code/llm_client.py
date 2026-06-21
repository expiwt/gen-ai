"""
LLM-клиент с поддержкой response_model, retry и структурированного вывода.

Использует OpenAI-совместимый API (DeepSeek, OpenAI, кастомные endpoint'ы).
Работает через JSON mode + Pydantic-валидацию (не требует structured outputs API).
"""
import logging
import os
import re
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _get_client() -> OpenAI:
    """Создать клиент с verify=False для self-signed сертификатов."""
    import httpx
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
        http_client=httpx.Client(verify=False),
    )


def llm_complete(
    system_prompt: str,
    user_prompt: str,
    response_model: Optional[Type[T]] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> tuple[str, dict]:
    """
    Вызвать LLM, опционально со структурированным выводом.

    Подход: JSON mode + инструкция в промпте -> Pydantic-валидация.
    Работает на DeepSeek, OpenAI и других без structured outputs API.

    Returns:
        (текст ответа, usage словарь)
    """
    client = _get_client()
    model_name = model or os.getenv("LLM_MODEL", "deepseek-chat")
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            kwargs = dict(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )

            # Если нужен структурированный вывод — просим JSON
            if response_model is not None:
                schema_hint = _build_json_schema_hint(response_model)
                messages[1]["content"] += (
                    f"\n\nОТВЕТЬ СТРОГО В JSON ПО ЭТОЙ СХЕМЕ:\n{schema_hint}"
                )
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)

            content = resp.choices[0].message.content or ""
            if resp.usage:
                usage_info = {
                    "prompt_tokens": resp.usage.prompt_tokens or 0,
                    "completion_tokens": resp.usage.completion_tokens or 0,
                    "total_tokens": (resp.usage.prompt_tokens or 0)
                                   + (resp.usage.completion_tokens or 0),
                }

            # Парсим JSON, если нужен response_model
            if response_model is not None:
                cleaned = _extract_json(content)
                try:
                    response_model.model_validate_json(cleaned)
                    return cleaned, usage_info
                except Exception as e:
                    logger.warning(
                        "Attempt %d/%d: parse fail for %s: %s | content: %s",
                        attempt + 1, max_retries,
                        response_model.__name__, e, content[:150],
                    )
                    if attempt == max_retries - 1:
                        return content, usage_info
                    continue

            return content, usage_info

        except Exception as e:
            logger.warning(
                "LLM call attempt %d/%d failed: %s",
                attempt + 1, max_retries, e,
            )
            if attempt == max_retries - 1:
                raise

    return "", usage_info


def _build_json_schema_hint(model_class: Type[BaseModel]) -> str:
    """Создать JSON Schema hint для промпта."""
    schema = model_class.model_json_schema()
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = ["{"]
    for name, prop in props.items():
        typ = prop.get("type", "string")
        desc = prop.get("description", "")
        req = "(required)" if name in required else "(optional)"
        enum_vals = prop.get("enum")
        if enum_vals:
            desc += f" один из: {enum_vals}"
        lines.append(f'  "{name}": {typ}  {req}  {desc}')
    lines.append("}")
    return "\n".join(lines)


def _extract_json(text: str) -> str:
    """Извлечь JSON-объект из текста (убрать markdown-обёртку)."""
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        return m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text
