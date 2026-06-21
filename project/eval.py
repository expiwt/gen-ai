"""
Eval: прогон пайплайна на 15 тестовых входах с замером метрик.

5 бумаг × 3 среза времени = 15 тестов.

Текущая версия — 5 бумаг с текущими данными + 2 дополнительных прогона
с модифицированными данными (симуляция разных дат).

Для каждого входа замеряем:
- Правильность (LLM-as-judge по 5 шкалам: 0–5)
- Путь (число шагов, токены, стоимость)
- Прошёл/не прошёл
"""
import csv
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Добавляем code/ в путь
sys.path.insert(0, str(Path(__file__).parent / "code"))

from moex_client import fetch_marketdata, fetch_marketdata_by_date
from rag import NewsCorpus
from pipeline import run_analysis


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Критерии оценки LLM-as-judge
EVAL_CRITERIA = """
Оцени анализ акции по 5 шкалам (каждая 0 или 1):

1. DATA_USAGE: Все ли MOEX-данные (цена, кап-ция, объём) корректно отражены?
2. CONSISTENCY: Нет внутренних противоречий (например, strong_buy при потенциале 2%)?
3. FACTUAL: Все утверждения подтверждены данными (нет выдуманных цифр)?
4. SPECIFICITY: Есть конкретика (цифры, проценты, направления), а не общие слова?
5. PLAUSIBILITY: Выводы выглядят разумными для данной бумаги?

Ответь строго в формате JSON:
{"score_1": 0/1, "score_2": 0/1, "score_3": 0/1, "score_4": 0/1, "score_5": 0/1, "comment": "..."}
"""


def _serialize(obj):
    """Сериализация для JSON: date/datetime → isoformat."""
    import datetime as _dt
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def judge_eval(analysis_dict: dict, market_data_dict: dict) -> dict:
    """LLM-as-judge оценивает качество ответа."""
    from llm_client import llm_complete

    prompt = f"""=== РЫНОЧНЫЕ ДАННЫЕ (эталон) ===
{json.dumps(market_data_dict, ensure_ascii=False, indent=2, default=_serialize)}

=== ОТВЕТ АНАЛИТИКА ===
{json.dumps(analysis_dict, ensure_ascii=False, indent=2, default=_serialize)}

{EVAL_CRITERIA}
"""
    content, usage = llm_complete(
        system_prompt="Ты строгий оценщик качества финансового анализа.",
        user_prompt=prompt,
        temperature=0,
        max_retries=2,
    )

    # Очищаем JSON от markdown-обёртки
    import re as _re
    _m = _re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
    _clean = _m.group(1) if _m else content
    try:
        result = json.loads(_clean)
        scores = [result.get(f"score_{i}", 0) for i in range(1, 6)]
        total = sum(scores)
        return {
            "correctness_score": total,
            "scores": {f"criterion_{i}": s for i, s in enumerate(scores, 1)},
            "comment": result.get("comment", ""),
        }
    except Exception:
        return {
            "correctness_score": 0,
            "scores": {},
            "comment": f"Failed to parse judge response: {content[:200]}",
        }


def run_eval(out_dir: str = "./output/eval"):
    """Запустить eval на всех тестовых входах."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Инициализация
    corpus = NewsCorpus()
    corpus.load("./input/news")

    # Тестовые входы: 5 бумаг × 3 сценария (реальные даты из истории MOEX)
    tickers = ["MAGN", "RAGR", "ALRS", "GMKN", "SVCB"]
    scenarios = [
        {"name": "latest", "label": "Последние данные (19.06)", "date": None},  # None = текущие
        {"name": "mid",    "label": "Две недели назад (05.06)", "date": "2026-06-05"},
        {"name": "early",  "label": "Месяц назад (18.05)",      "date": "2026-05-18"},
    ]

    results = []
    row_id = 0

    for ticker in tickers:
        for scenario in scenarios:
            row_id += 1
            logger.info(
                "[%d/15] %s — %s", row_id, ticker, scenario["label"]
            )

            # Получаем реальные данные на дату
            if scenario["date"] is None:
                md = fetch_marketdata(ticker)
            else:
                md = fetch_marketdata_by_date(ticker, scenario["date"])

            if md is None:
                logger.warning("Skipping %s — %s: no MOEX data", ticker, scenario["label"])
                results.append({
                    "ticker": ticker,
                    "scenario": scenario["name"],
                    "passed": False,
                    "correctness_score": 0,
                    "steps": 0,
                    "tokens": 0,
                    "cost_usd": 0,
                    "hallu_passed": False,
                    "hallu_errors": ["MOEX data fetch failed"],
                    "judge_comment": "",
                    "growth_outlook": "",
                    "growth_potential": 0,
                    "risk_level": "",
                })
                continue

            analysis, meta = run_analysis(ticker, corpus, market_data=md)

            if analysis is None:
                results.append({
                    "ticker": ticker,
                    "scenario": scenario["name"],
                    "passed": False,
                    "correctness_score": 0,
                    "steps": meta.get("steps", 0),
                    "tokens": meta.get("total_tokens", 0),
                    "cost_usd": meta.get("cost_usd", 0),
                    "errors": meta.get("errors", ["Analysis failed"]),
                    "hallu_passed": False,
                })
                continue

            # LLM-as-judge
            judge_result = judge_eval(
                analysis.model_dump(),
                md.model_dump(),
            )

            passed = (
                judge_result["correctness_score"] >= 3
                and analysis.hallu_check_passed
            )

            row = {
                "ticker": ticker,
                "scenario": scenario["name"],
                "passed": passed,
                "correctness_score": judge_result["correctness_score"],
                "steps": meta.get("steps", 0),
                "tokens": meta.get("total_tokens", 0),
                "cost_usd": meta.get("cost_usd", 0),
                "hallu_passed": analysis.hallu_check_passed,
                "hallu_errors": meta.get("hallu_check", {}).get("errors", []),
                "judge_comment": judge_result.get("comment", ""),
                "growth_outlook": analysis.growth_outlook,
                "growth_potential": analysis.growth_potential_percent,
                "risk_level": analysis.risk_level,
            }
            results.append(row)

    # Сохраняем результаты
    eval_path = out_path / "eval_results.csv"
    if results:
        with open(eval_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info("Saved eval results to %s", eval_path)

    # Сохраняем JSON
    json_path = out_path / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Saved eval results to %s", json_path)

    # Итоговая статистика
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    avg_score = sum(r["correctness_score"] for r in results) / max(total, 1)
    avg_tokens = sum(r["tokens"] for r in results) / max(total, 1)
    total_cost = sum(r["cost_usd"] for r in results)

    summary = {
        "total_tests": total,
        "passed": passed_count,
        "pass_rate": f"{passed_count}/{total} ({passed_count/max(total,1)*100:.0f}%)",
        "avg_correctness_score": round(avg_score, 2),
        "avg_tokens_per_run": round(avg_tokens, 0),
        "total_cost_usd": round(total_cost, 4),
        "hallu_check_pass_rate": f"{sum(1 for r in results if r['hallu_passed'])}/{total}",
    }

    summary_path = out_path / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Summary: %s", json.dumps(summary, ensure_ascii=False))

    return results, summary


if __name__ == "__main__":
    results, summary = run_eval()
