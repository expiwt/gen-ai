"""
Генерация графиков по результатам eval.

Читает output/eval/eval_results.csv и рисует 3 графика:
1. Pass rate по тикерам
2. Avg correctness score по тикерам
3. Tokens per run по тикерам

Сохраняет в output/eval/charts/
"""
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# Настройка русских шрифтов — DejaVu Sans, есть везде
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

CHART_DIR = Path(__file__).parent.parent / "output" / "eval" / "charts"


def load_results(csv_path: str) -> list[dict]:
    """Загрузить результаты из CSV."""
    results = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["passed"] = row.get("passed", "False") == "True"
            row["correctness_score"] = float(row.get("correctness_score", 0))
            row["total_tokens"] = int(float(row.get("total_tokens", 0)))
            results.append(row)
    return results


def chart_pass_rate_by_ticker(results: list[dict]):
    """График 1: pass rate по тикерам."""
    ticker_data = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        t = r["ticker"]
        ticker_data[t]["total"] += 1
        if r["passed"]:
            ticker_data[t]["passed"] += 1

    tickers = sorted(ticker_data.keys())
    rates = [
        ticker_data[t]["passed"] / max(ticker_data[t]["total"], 1) * 100
        for t in tickers
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4CAF50" if r >= 50 else "#f44336" for r in rates]
    bars = ax.bar(tickers, rates, color=colors, width=0.7)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Pass rate по тикерам", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.3, label="50%")
    ax.legend()
    plt.tight_layout()
    save_path = CHART_DIR / "pass_rate_by_ticker.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранён: {save_path}")


def chart_avg_correctness_by_ticker(results: list[dict]):
    """График 2: avg correctness score по тикерам."""
    ticker_data = defaultdict(lambda: {"total": 0, "sum_score": 0.0})
    for r in results:
        t = r["ticker"]
        ticker_data[t]["total"] += 1
        ticker_data[t]["sum_score"] += r["correctness_score"]

    tickers = sorted(ticker_data.keys())
    avgs = [
        ticker_data[t]["sum_score"] / max(ticker_data[t]["total"], 1)
        for t in tickers
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3"] * len(tickers)
    bars = ax.bar(tickers, avgs, color=colors, width=0.7)

    for bar, avg in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{avg:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Avg correctness score (0-5)")
    ax.set_title("Средняя оценка правильности по тикерам", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3, color="orange", linestyle="--", alpha=0.3, label="Порог прохода")
    ax.legend()
    plt.tight_layout()
    save_path = CHART_DIR / "avg_correctness_by_ticker.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранён: {save_path}")


def chart_tokens_by_ticker(results: list[dict]):
    """График 3: среднее количество токенов по тикерам."""
    ticker_data = defaultdict(lambda: {"total": 0, "sum_tokens": 0})
    for r in results:
        t = r["ticker"]
        ticker_data[t]["total"] += 1
        ticker_data[t]["sum_tokens"] += r["total_tokens"]

    tickers = sorted(ticker_data.keys())
    avgs = [
        ticker_data[t]["sum_tokens"] / max(ticker_data[t]["total"], 1)
        for t in tickers
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#FF9800"] * len(tickers)
    bars = ax.bar(tickers, avgs, color=colors, width=0.7)

    for bar, avg in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{avg:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Avg tokens per run")
    ax.set_title("Среднее количество токенов на прогон по тикерам", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = CHART_DIR / "tokens_by_ticker.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранён: {save_path}")


def main():
    csv_path = Path(__file__).parent.parent / "output" / "eval" / "eval_results.csv"
    if not csv_path.exists():
        print(f"Файл не найден: {csv_path}")
        print("Сначала запусти eval.py")
        return

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results(str(csv_path))
    print(f"Загружено {len(results)} результатов")

    chart_pass_rate_by_ticker(results)
    chart_avg_correctness_by_ticker(results)
    chart_tokens_by_ticker(results)

    print(f"\nВсе графики сохранены в {CHART_DIR}/")


if __name__ == "__main__":
    main()
