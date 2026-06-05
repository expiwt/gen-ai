"""
schema.py — Pydantic-модели для анализа экспертных интервью (Вариант B).

Адаптация семинарского пайплайна под предметную область:
- Participant → Expert
- concerns → claims (тезисы)
- Аспекты: новизна, обоснованность, практичность, риски
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════
# Раунд 1: Information Extraction — сущности из текста
# ═══════════════════════════════════════════════════

EXPERTISE_LEVELS = [
    "академический_учёный",
    "практик_отрасли",
    "журналист_аналитик",
    "госслужащий",
    "предприниматель",
    "неизвестно",
]

TOPIC_AREAS = [
    "экономика",
    "геополитика",
    "банковская_система",
    "технологии",
    "социальная_политика",
    "энергетика",
    "финансовые_рынки",
]


class Claim(BaseModel):
    """Один тезис, высказанный экспертом."""
    claim_id: int = Field(..., description="Уникальный номер тезиса")
    text: str = Field(..., description="Текст тезиса (прямая или косвенная цитата)")
    topic: str = Field(..., description="Тема тезиса")
    is_forecast: bool = Field(..., description="Является ли тезис прогнозом")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Уверенность эксперта (если выражена явно)"
    )


class Expert(BaseModel):
    """Эксперт, дающий интервью."""
    name: str = Field(..., description="Имя эксперта")
    expertise_level: str = Field(
        ..., description="Уровень экспертизы",
    )
    topic_area: str = Field(
        ..., description="Основная тематическая область",
    )
    claims: list[Claim] = Field(..., description="Список тезисов")

    @field_validator("expertise_level")
    @classmethod
    def must_be_known_level(cls, v: str) -> str:
        if v not in EXPERTISE_LEVELS:
            raise ValueError(
                f"Уровень «{v}» не из списка: {EXPERTISE_LEVELS}"
            )
        return v

    @field_validator("topic_area")
    @classmethod
    def must_be_known_topic(cls, v: str) -> str:
        if v not in TOPIC_AREAS:
            raise ValueError(
                f"Тема «{v}» не из списка: {TOPIC_AREAS}"
            )
        return v


# ═══════════════════════════════════════════════════
# Раунд 2: Аспектный анализ
# ═══════════════════════════════════════════════════

ASPECTS = [
    "новизна",
    "обоснованность",
    "практичность",
    "реалистичность",
]


class AspectScore(BaseModel):
    """Оценка по одному аспекту."""
    aspect: str = Field(..., description="Название аспекта")
    score: int = Field(..., ge=1, le=5, description="Оценка от 1 до 5")
    reasoning: str = Field(..., description="Обоснование оценки")
    quote: str = Field(..., description="Цитата-подтверждение из текста")

    @field_validator("aspect")
    @classmethod
    def must_be_known_aspect(cls, v: str) -> str:
        if v not in ASPECTS:
            raise ValueError(f"Аспект «{v}» не из списка: {ASPECTS}")
        return v


class AspectAnalysis(BaseModel):
    """Результат аспектного анализа одного интервью."""
    expert_name: str = Field(..., description="Имя эксперта")
    claims_evaluated: list[AspectScore] = Field(
        ..., description="Оценки по аспектам для каждого тезиса"
    )


# ═══════════════════════════════════════════════════
# Раунд 3: Map-Reduce — свёртка
# ═══════════════════════════════════════════════════

class ReducedSummary(BaseModel):
    """Итоговая сводка после Map-Reduce."""
    title: str = Field(..., description="Название сводки")
    main_claims: list[str] = Field(
        ..., description="Ключевые тезисы (до 5)"
    )
    average_scores: dict[str, float] = Field(
        ..., description="Средние оценки по каждому аспекту"
    )
    total_claims_analyzed: int = Field(
        ..., ge=0, description="Всего проанализировано тезисов"
    )
    consensus_level: str = Field(
        ...,
        description="Уровень согласованности тезисов",
    )


# ═══════════════════════════════════════════════════
# Раунд 5: LLM-as-judge
# ═══════════════════════════════════════════════════

class AutodiscoveryResult(BaseModel):
    """Результат autodiscovery — дополнительные аспекты."""
    additional_aspects: list[dict] = Field(
        ...,
        description="Список дополнительных аспектов: [{\"name\": \"...\", \"reason\": \"...\", \"example\": \"...\"}]"
    )
    """Оценка качества работы пайплайна."""
    overall_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Общая оценка от 0 до 1"
    )
    completeness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Полнота охвата тезисов"
    )
    factuality: float = Field(
        ..., ge=0.0, le=1.0,
        description="Фактологическая точность"
    )
    consistency: float = Field(
        ..., ge=0.0, le=1.0,
        description="Согласованность сводки"
    )
    action_items: list[dict] = Field(
        ...,
        description="Что улучшить: [{\"item\": \"...\", \"severity\": \"high|medium|low\"}]"
    )
    verdict: str = Field(
        ..., description="Вердикт судьи (до 3 предложений)"
    )


# ═══════════════════════════════════════════════════
# Утилита: check_quotes — проверка галлюцинаций
# ═══════════════════════════════════════════════════

def check_quotes(
    claims: list[Claim],
    source_text: str,
    min_overlap: int = 10,
) -> tuple[list[Claim], list[Claim]]:
    """
    Проверяет, что каждый тезис имеет опору в исходном тексте.

    Args:
        claims: список извлечённых тезисов
        source_text: исходный текст интервью
        min_overlap: минимальное количество совпадающих символов

    Returns:
        (valid_claims, ghost_claims) — валидные и галлюцинированные тезисы
    """
    source_lower = source_text.lower()
    valid: list[Claim] = []
    ghost: list[Claim] = []

    for claim in claims:
        # Ищем самый длинный общий подотрезок между claim.text и source_text
        claim_lower = claim.text.lower()
        # Разбиваем на слова и ищем их вхождение
        words = re.findall(r'\w+', claim_lower)
        if not words:
            ghost.append(claim)
            continue

        # Считаем, сколько слов из тезиса встречаются в исходнике
        found = sum(1 for w in words if w in source_lower)
        overlap_ratio = found / len(words) if words else 0

        if overlap_ratio >= 0.5 or found >= min_overlap:
            valid.append(claim)
        else:
            ghost.append(claim)

    return valid, ghost
class JudgeScore(BaseModel):
    """Оценка качества работы пайплайна."""
    overall_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Общая оценка от 0 до 1"
    )
    completeness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Полнота охвата тезисов"
    )
    factuality: float = Field(
        ..., ge=0.0, le=1.0,
        description="Фактологическая точность"
    )
    consistency: float = Field(
        ..., ge=0.0, le=1.0,
        description="Согласованность сводки"
    )
    action_items: list[dict] = Field(
        ...,
        description="Что улучшить: [{\"item\": \"...\", \"severity\": \"high|medium|low\"}]"
    )
    verdict: str = Field(
        ..., description="Вердикт судьи (до 3 предложений)"
    )
