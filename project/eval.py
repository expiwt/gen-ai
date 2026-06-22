"""
Eval: прогон пайплайна на 50+ тестовых входах с замером метрик.

15 тикеров x 5-6 исторических дат = 75+ тестов.

Сохраняет результаты инкрементально (после каждого тикера).
"""
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

from moex_client import fetch_marketdata, fetch_marketdata_by_date
from rag import NewsCorpus
from pipeline import run_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Критерии оценки LLM-as-judge
EVAL_CRITERIA = """
Оцени анализ акции по 5 шкалам (каждая 0 или 1):

1. DATA_USAGE: Все ли MOEX-данные (цена, кап-ция, объём) корректно отражены?
2. CONSISTENCY: Нет внутренних противоречий?
3. FACTUAL: Все утверждения подтверждены данными (нет выдуманных цифр)?
4. SPECIFICITY: Есть конкретика (цифры, проценты, направления)?
5. PLAUSIBILITY: Выводы выглядят разумными для данной бумаги?

Ответь строго в формате JSON:
{"score_1": 0/1, "score_2": 0/1, "score_3": 0/1, "score_4": 0/1, "score_5": 0/1, "comment": "..."}
"""


def _serialize(obj):
    import datetime as _dt
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def judge_eval(analysis_dict: dict, market_data_dict: dict) -> dict:
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

    import re as _re
    _m = _re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
    _clean = _m.group(1) if _m else content
    try:
        result = json.loads(_clean)
        scores = [result.get(f"score_{i}", 0) for i in range(1, 6)]
        total = sum(scores)
        return {"correctness_score": total, "comment": result.get("comment", "")}
    except Exception:
        return {"correctness_score": 0, "comment": f"Parse error: {content[:200]}"}


def save_intermediate(results: list[dict], summary_path: Path, out_path: Path):
    """Сохранить промежуточные результаты."""
    # CSV
    if results:
        fieldnames = [
            "ticker", "scenario", "passed", "correctness_score",
            "hallu_passed", "hallu_errors",
            "steps", "prompt_tokens", "completion_tokens", "total_tokens",
            "cost_usd", "latency_sec",
        ]
        csv_path = out_path / "eval_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # JSON
    json_path = out_path / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    avg_score = sum(r["correctness_score"] for r in results) / max(total, 1)
    avg_tokens = sum(r["total_tokens"] for r in results) / max(total, 1)
    total_cost = sum(r["cost_usd"] for r in results)
    hallu_passed_count = sum(1 for r in results if r["hallu_passed"])
    avg_latency = sum(r["latency_sec"] for r in results) / max(total, 1)

    summary = {
        "total_tests": total,
        "passed": passed_count,
        "pass_rate": f"{passed_count}/{total} ({passed_count/max(total,1)*100:.0f}%)",
        "avg_correctness_score": round(avg_score, 2),
        "avg_tokens_per_run": round(avg_tokens, 0),
        "total_cost_usd": round(total_cost, 4),
        "hallu_check_pass_rate": f"{hallu_passed_count}/{total}",
        "avg_latency_sec": round(avg_latency, 2),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def run_eval(out_dir: str = "./output/eval"):
    """Запустить eval на всех тестовых входах."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    corpus = NewsCorpus()
    corpus.load("./input/news")

    # 15 тикеров
    tickers = [
        "GAZP", "SBER", "LKOH", "ROSN", "VTBR",
        "TATN", "NLMK", "SNGS", "PIKK", "MGNT",
        "MAGN", "RAGR", "ALRS", "GMKN", "SVCB",
    ]

    weekly_dates = [
        "2026-04-01", "2026-04-07", "2026-04-14", "2026-04-21", "2026-04-28",
        "2026-05-05", "2026-05-12", "2026-05-18", "2026-05-25",
        "2026-06-01", "2026-06-08", "2026-06-15", "2026-06-19",
    ]

    scenarios = []
    for d in weekly_dates:
        label = d.replace("2026-", "")
        scenarios.append({"name": d, "label": f"Дата {label}", "date": d})
    scenarios.append({"name": "latest", "label": "Последние данные", "date": None})

    results = []
    row_id = 0
    total_planned = len(tickers) * len(scenarios)
    logger.info("Всего запланировано: %d прогонов (%d тикеров x %d дат)",
                total_planned, len(tickers), len(scenarios))

    for ticker in tickers:
        for scenario in scenarios:
            row_id += 1

            t_start = time.perf_counter()

            if scenario["date"] is None:
                md = fetch_marketdata(ticker)
            else:
                md = fetch_marketdata_by_date(ticker, scenario["date"])

            if md is None:
                logger.info("[%d/%d] %s — %s: нет MOEX данных (выходной/таймаут)",
                            row_id, total_planned, ticker, scenario["label"])
                results.append({
                    "ticker": ticker, "scenario": scenario["name"],
                    "passed": False, "correctness_score": 0,
                    "hallu_passed": False, "hallu_errors": "MOEX data fetch failed",
                    "steps": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "cost_usd": 0, "latency_sec": 0,
                })
                continue

            # Один print для прогресса
            sys.stdout.write(f"\r  [{row_id}/{total_planned}] {ticker} — {scenario['label']} ... ")
            sys.stdout.flush()

            analysis, meta = run_analysis(ticker, corpus, market_data=md)
            latency_sec = round(time.perf_counter() - t_start, 2)

            if analysis is None:
                results.append({
                    "ticker": ticker, "scenario": scenario["name"],
                    "passed": False, "correctness_score": 0,
                    "hallu_passed": False,
                    "hallu_errors": str(meta.get("errors", ["Analysis failed"])),
                    "steps": meta.get("steps", 0),
                    "prompt_tokens": meta.get("prompt_tokens", 0),
                    "completion_tokens": meta.get("completion_tokens", 0),
                    "total_tokens": meta.get("total_tokens", 0),
                    "cost_usd": round(meta.get("cost_usd", 0), 6),
                    "latency_sec": latency_sec,
                })
                sys.stdout.write(f"анализ не удался\n")
                sys.stdout.flush()
                continue

            judge_result = judge_eval(analysis.model_dump(), md.model_dump())
            passed = (judge_result["correctness_score"] >= 3 and analysis.hallu_check_passed)
            hallu_errors = meta.get("hallu_check", {}).get("errors", [])

            results.append({
                "ticker": ticker, "scenario": scenario["name"],
                "passed": passed,
                "correctness_score": judge_result["correctness_score"],
                "hallu_passed": analysis.hallu_check_passed,
                "hallu_errors": "; ".join(hallu_errors) if hallu_errors else "",
                "steps": meta.get("steps", 0),
                "prompt_tokens": meta.get("prompt_tokens", 0),
                "completion_tokens": meta.get("completion_tokens", 0),
                "total_tokens": meta.get("total_tokens", 0),
                "cost_usd": round(meta.get("cost_usd", 0), 6),
                "latency_sec": latency_sec,
            })
            sys.stdout.write(f"✓\n")
            sys.stdout.flush()

        # Сохраняем после каждого тикера
        save_intermediate(results, out_path / "eval_summary.json", out_path)

    return results, json.load(open(out_path / "eval_summary.json"))


if __name__ == "__main__":
    results, summary = run_eval()
    print("\nEval завершён!")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
