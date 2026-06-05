"""
pipeline.py — Конвейер анализа экспертных интервью.

Этапы:
1. IE: извлечение сущностей (Expert, Claims)
2. Аспектный анализ: оценка тезисов по аспектам
3. Map-Reduce: свёртка результатов
4. LLM-as-judge: оценка качества
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Корень проекта
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from schema import (
    EXPERTISE_LEVELS,
    TOPIC_AREAS,
    ASPECTS,
    Expert,
    Claim,
    AspectAnalysis,
    AspectScore,
    ReducedSummary,
    JudgeScore,
    check_quotes,
)
from prompts import (
    IE_SYSTEM_PROMPT,
    IE_USER_PROMPT,
    ASPECT_SYSTEM_PROMPT,
    ASPECT_USER_PROMPT,
    REDUCE_SYSTEM_PROMPT,
    REDUCE_USER_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
)

# Клиент — из семинарной фабрики
sys.path.insert(0, str(Path(__file__).parents[1] / "семинар_2" / "starter"))
from llm_client import make_client, get_model

client = make_client()
MODEL = get_model()


def analyze(input_path: str) -> dict:
    """
    Главная функция пайплайна.

    Args:
        input_path: путь к папке с файлами интервью или к одному файлу

    Returns:
        Словарь со всеми артефактами
    """
    start_time = time.time()
    total_tokens = 0

    
    # Собираем все входные файлы
    path = Path(input_path)
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.txt"))

    print(f"Найдено файлов: {len(files)}")
    print(f"Модель: {MODEL}")

    # Раунд 1: Information Extraction
    print("Раунд 1: Information Extraction")

    all_experts: list[Expert] = []
    source_texts: dict[str, str] = {}

    for filepath in files:
        filename = filepath.name
        text = filepath.read_text(encoding="utf-8")
        source_texts[filename] = text

        print(f"\n--- {filename} ---")

        ie_prompt = IE_SYSTEM_PROMPT.format(
            expertise_levels=EXPERTISE_LEVELS,
            topic_areas=TOPIC_AREAS,
        )

        try:
            expert, completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": ie_prompt},
                    {"role": "user", "content": IE_USER_PROMPT.format(text=text)},
                ],
                response_model=Expert,
                max_retries=3,
                with_completion=True,
            )
            total_tokens += completion.usage.total_tokens if completion.usage else 0
            all_experts.append(expert)
            print(f"  Эксперт: {expert.name}")
            print(f"  Тезисов: {len(expert.claims)}")

            # Проверка галлюцинаций
            valid, ghost = check_quotes(expert.claims, text)
            print(f"  Валидных: {len(valid)}, Ghost-цитат: {len(ghost)}")

        except Exception as e:
            print(f"  Ошибка IE: {e}")

    # Сохраняем IE-результат
    ie_output = []
    for expert in all_experts:
        ie_output.append(expert.model_dump())
    (PROJECT_ROOT / "output" / "participants.json").write_text(
        json.dumps(ie_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n IE сохранён: output/participants.json")

    # Раунд 2: Аспектный анализ
    print("Раунд 2: Аспектный анализ")


    all_aspects: list[AspectAnalysis] = []

    for expert in all_experts:
        claims_text = "\n".join(
            f"{c.claim_id}. [{c.topic}] {c.text}"
            for c in expert.claims
        )

        try:
            aspect_result, completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": ASPECT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": ASPECT_USER_PROMPT.format(
                            expert_name=expert.name,
                            claims=claims_text,
                        ),
                    },
                ],
                response_model=AspectAnalysis,
                max_retries=3,
                with_completion=True,
            )
            total_tokens += completion.usage.total_tokens if completion.usage else 0
            all_aspects.append(aspect_result)
            print(f"  {expert.name}: {len(aspect_result.claims_evaluated)} оценок")

        except Exception as e:
            print(f"  Ошибка аспектного анализа: {e}")

    # Сохраняем аспекты
    aspects_output = [a.model_dump() for a in all_aspects]
    (PROJECT_ROOT / "output" / "aspects.json").write_text(
        json.dumps(aspects_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f" Аспекты сохранены: output/aspects.json")

    # Раунд 2.5: Autodiscovery

    print("Раунд 2.5: Autodiscovery (динамические аспекты)")

    from prompts import AUTODISCOVERY_PROMPT

    # Собираем все тезисы для анализа
    all_claims_text = []
    for expert in all_experts:
        for c in expert.claims:
            all_claims_text.append(f"{c.claim_id}. {c.text}")
    claims_str = "\n".join(all_claims_text[:10])  # первые 10 для контекста

    try:
        from schema import AutodiscoveryResult
        auto_result, completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": AUTODISCOVERY_PROMPT.format(
                    aspects=ASPECTS
                )},
                {"role": "user", "content": f"Тезисы интервью:\n{claims_str}"},
            ],
            response_model=AutodiscoveryResult,
            max_retries=2,
            with_completion=True,
        )
        # Сохраняем
        (PROJECT_ROOT / "output" / "autodiscovery.json").write_text(
            json.dumps(auto_result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f" Autodiscovery сохранён: output/autodiscovery.json")
        # Выводим предложенные аспекты
        for asp in auto_result.additional_aspects:
            print(f"  • {asp.get('name', '?')}: {asp.get('reason', '')[:80]}...")
    except Exception as e:
        print(f"  Autodiscovery пропущен (не критично): {e}")
        auto_response = None

    # Раунд 3: Map-Reduce
    print("Раунд 3: Map-Reduce")

    # Собираем все оценки для редуцирования
    all_scores = []
    for analysis in all_aspects:
        for score in analysis.claims_evaluated:
            all_scores.append(score.model_dump())

    aspects_data_str = json.dumps(all_scores, ensure_ascii=False, indent=2)

    try:
        summary, completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": REDUCE_SYSTEM_PROMPT},
                {"role": "user", "content": REDUCE_USER_PROMPT.format(
                    aspects_data=aspects_data_str
                )},
            ],
            response_model=ReducedSummary,
            max_retries=3,
            with_completion=True,
        )
        total_tokens += completion.usage.total_tokens if completion.usage else 0
        print(f"  Заголовок: {summary.title}")
        print(f"  Тезисов: {summary.total_claims_analyzed}")
        print(f"  Согласованность: {summary.consensus_level}")

    except Exception as e:
        print(f"  Ошибка Reduce: {e}")
        # Fallback — подсчёт средних вручную
        avg_scores = {}
        for aspect in ASPECTS:
            scores = [s["score"] for s in all_scores if s["aspect"] == aspect]
            avg_scores[aspect] = sum(scores) / len(scores) if scores else 0.0

        summary = ReducedSummary(
            title="Автоматическая сводка",
            main_claims=[],
            average_scores=avg_scores,
            total_claims_analyzed=len(all_scores),
            consensus_level="не определён",
        )

    # Сохраняем summary
    (PROJECT_ROOT / "output" / "summary.json").write_text(
        json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f" Summary сохранён: output/summary.json")

    # Раунд 5: LLM-as-Judge
    print("Раунд 5: LLM-as-Judge")


    source_info = f"{len(files)} интервью, {len(all_experts)} экспертов"

    try:
        judge_result, completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": JUDGE_USER_PROMPT.format(
                        source_info=source_info,
                        summary=json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
                    ),
                },
            ],
            response_model=JudgeScore,
            max_retries=3,
            with_completion=True,
        )
        total_tokens += completion.usage.total_tokens if completion.usage else 0

    except Exception as e:
        print(f"  Ошибка Judge: {e}")
        judge_result = JudgeScore(
            overall_score=0.5,
            completeness=0.5,
            factuality=0.5,
            consistency=0.5,
            action_items=[{"item": "ручная проверка", "severity": "high"}],
            verdict="Ошибка при вызове judge",
        )

    # Сохраняем judge
    (PROJECT_ROOT / "output" / "judge_report.json").write_text(
        json.dumps(judge_result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Если оценка < 0.7 — повторяем Reduce с улучшенным промптом
    if judge_result.overall_score < 0.7:
        print(f"\n⚠ overall_score = {judge_result.overall_score} < 0.7")
        print("→ Повторный Reduce с улучшенным промптом...")
        
        improved_reduce_prompt = """Ты — аналитик, составляющий итоговую сводку.

Все фрагменты — одно интервью с Михаилом Хазиным.

Требования к сводке:
- Выдели 4-5 ключевых тезисов (главные идеи интервью)
- average_scores: посчитай как СРЕДНЕЕ АРИФМЕТИЧЕСКОЕ всех оценок по каждому аспекту
- total_claims_analyzed: точное число проанализированных тезисов
- consensus_level: если средние оценки 3.0+ — "высокий", 2.0-2.9 — "средний", <2.0 — "низкий"

Сводка должна быть:
- Полной (охватить все основные темы интервью: экономика, геополитика, банки, прогнозы)
- Фактологически точной (только то, что сказано)
- Структурированной

Верни ТОЛЬКО JSON."""
        
        try:
            summary, completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": improved_reduce_prompt},
                    {"role": "user", "content": REDUCE_USER_PROMPT.format(
                        aspects_data=aspects_data_str
                    )},
                ],
                response_model=ReducedSummary,
                max_retries=3,
                with_completion=True,
            )
            (PROJECT_ROOT / "output" / "summary.json").write_text(
                json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f" Улучшенный summary сохранён")
            
            # Перезапускаем Judge на улучшенном summary
            print("→ Перезапуск Judge на улучшенном summary...")
            try:
                judge_result, completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": JUDGE_USER_PROMPT.format(
                                source_info=source_info,
                                summary=json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
                            ),
                        },
                    ],
                    response_model=JudgeScore,
                    max_retries=3,
                    with_completion=True,
                )
                (PROJECT_ROOT / "output" / "judge_report.json").write_text(
                    json.dumps(judge_result.model_dump(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f" Judge перезапущен: overall_score = {judge_result.overall_score}")
            except Exception as e:
                print(f"  Ошибка перезапуска Judge: {e}")
        except Exception as e:
            print(f"  Ошибка повторного Reduce: {e}")

    print(f"\n Judge отчёт сохранён: output/judge_report.json")
    print(f"  overall_score: {judge_result.overall_score}")
    print(f"  verdict: {judge_result.verdict}")

    # Итоги
    
    elapsed = time.time() - start_time
    cost_estimate = (total_tokens / 1_000_000) * 2.0  # ~$2/M токенов

    result = {
        "stats": {
            "files": len(files),
            "experts": len(all_experts),
            "total_claims": sum(len(e.claims) for e in all_experts),
            "total_tokens": total_tokens,
            "time_seconds": round(elapsed, 1),
            "cost_estimate_usd": round(cost_estimate, 4),
        },
        "summary": summary.model_dump(),
        "judge": judge_result.model_dump(),
    }

    print(f"  Файлов: {result['stats']['files']}")
    print(f"  Экспертов: {result['stats']['experts']}")
    print(f"  Тезисов: {result['stats']['total_claims']}")
    print(f"  Токенов: {result['stats']['total_tokens']}")
    print(f"  Время: {result['stats']['time_seconds']} сек")
    print(f"  Стоимость: ${result['stats']['cost_estimate_usd']}")
    print(f"  Judge score: {result['stats']['cost_estimate_usd']}")

    return result


if __name__ == "__main__":
    input_dir = PROJECT_ROOT / "input"
    if not input_dir.exists():
        print(f"Папка {input_dir} не найдена")
        sys.exit(1)

    result = analyze(str(input_dir))

    # Сохраняем полный результат
    (PROJECT_ROOT / "output" / "full_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n Полный результат: output/full_result.json")
