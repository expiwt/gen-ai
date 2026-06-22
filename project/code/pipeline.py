"""
Основной пайплайн: агент, который анализирует акцию.

Поток:
1. Получить данные с MOEX (agent tool)
2. Загрузить новости, найти релевантные (RAG)
3. LLM анализирует данные + новости → StockAnalysis
4. Проверка галлюцинаций (LLM-as-judge)
"""
import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from llm_client import llm_complete
from moex_client import fetch_marketdata
import rag
from rag import NewsCorpus
from schema import EvalResult, MarketData, NewsArticle, StockAnalysis

load_dotenv()
logger = logging.getLogger(__name__)


# ---- Системный промпт для аналитика ----

ANALYST_SYSTEM_PROMPT = """Ты — профессиональный финансовый аналитик. Твоя задача — проанализировать акцию 
и дать структурированную оценку.

Твои данные:
1. Рыночные данные (цена, капитализация, объёмы, изменение) — ОБЯЗАТЕЛЬНО используй их
2. Новости по бумаге — используй для контекста

Правила:
- ВСЕ цифры и факты должны быть основаны ТОЛЬКО на предоставленных данных
- НЕ выдумывай новости, цифры или события
- Если данных недостаточно для уверенного вывода — отметь это в reasoning
- growth_outlook: strong_buy (>15% потенциал), buy (5-15%), hold (-5-5%), sell (-15- -5%), strong_sell (<-15%)
- risk_level: низкий/средний/высокий
- news_sentiment: от -1.0 (очень негативно) до +1.0 (очень позитивно)
- growth_potential_percent: от -50% до +200%
- top_news_themes: выдели до 3 ключевых тем из новостей
- market_cap_bln_rub: капитализация в МИЛЛИАРДАХ рублей (раздели на 1e9)
- daily_volume_mln_rub: объём в МИЛЛИОНАХ рублей (раздели на 1e6)
"""


def run_analysis(
    ticker: str,
    news_corpus: NewsCorpus,
    market_data: Optional[MarketData] = None,
) -> tuple[Optional[StockAnalysis], dict]:
    """
    Запустить полный анализ одной акции.

    Returns:
        (StockAnalysis или None, мета-информация со стоимостью и шагами)
    """
    meta = {
        "ticker": ticker,
        "steps": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "errors": [],
    }

    # Шаг 1: Получить данные MOEX
    logger.info("Step 1: Fetching MOEX data for %s", ticker)
    if market_data is None:
        market_data = fetch_marketdata(ticker)
    meta["steps"] += 1

    if market_data is None:
        meta["errors"].append("MOEX data fetch failed")
        return None, meta

    # Шаг 2: Найти релевантные новости
    logger.info("Step 2: Searching news for %s", ticker)
    news_results = news_corpus.search_by_ticker(ticker, top_k=3)
    meta["steps"] += 1

    top_news = [article for article, score in news_results]
    if top_news:
        rag_mode = "embeddings" if rag.EMBEDDING_AVAILABLE else "keyword"
        top_titles = " | ".join(f"{a.title} (score={s:.2f})" for a, s in news_results[:3])
        print(f"  RAG [{rag_mode}]: {len(news_results)} matches — {top_titles}")
        logger.info("RAG [%s] for %s: %d matches, top: %s", rag_mode, ticker, len(news_results),
                    " | ".join(f"{a.title} (score={s:.2f})" for a, s in news_results[:3]))
    else:
        logger.warning("RAG: no news found for %s", ticker)

    # Если новостей нет — создаём заглушку
    if not top_news:
        top_news = [NewsArticle(
            title="Нет новостей",
            source="",
            snippet="Новости по данной бумаге не найдены.",
        )]

    # Шаг 3: LLM-анализ
    logger.info("Step 3: Running LLM analysis for %s", ticker)
    user_prompt = _build_analysis_prompt(market_data, top_news)
    meta["steps"] += 1

    content, usage = llm_complete(
        system_prompt=ANALYST_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=StockAnalysis,
        temperature=0.1,
        max_retries=3,
    )

    meta["prompt_tokens"] += usage.get("prompt_tokens", 0)
    meta["completion_tokens"] += usage.get("completion_tokens", 0)
    meta["total_tokens"] += usage.get("total_tokens", 0)

    # Парсим ответ
    analysis = None
    try:
        analysis = StockAnalysis.model_validate_json(content)
        meta["steps"] += 1
    except Exception as e:
        err_msg = str(e)
        meta["errors"].append(f"Failed to parse StockAnalysis: {e}")
        logger.error("Parse error for %s: %s", ticker, e)

        # Попытка восстановления: если ошибка только в дате — фиксим
        if "analysis_date" in err_msg and "не может быть в будущем" in err_msg:
            try:
                import re as _re
                import json as _json
                raw = _json.loads(content)
                bad_date = str(raw.get("analysis_date", "???"))
                raw["analysis_date"] = str(date.today())
                repaired = _json.dumps(raw, ensure_ascii=False)
                analysis = StockAnalysis.model_validate_json(repaired)
                meta["steps"] += 1
                meta["hallu_check"] = {
                    "passed": False,
                    "errors": [f"analysis_date исправлена: было '{bad_date}' → сегодня"],
                }
                logger.info("Fixed hallucinated date for %s", ticker)
            except Exception as e2:
                meta["errors"].append(f"Date repair also failed: {e2}")

    # Шаг 4: Проверка галлюцинаций
    if analysis is not None:
        logger.info("Step 4: Running hallucination check for %s", ticker)
        meta["steps"] += 1
        hallu_result = _check_hallucinations(analysis, market_data, top_news)
        analysis.hallu_check_passed = hallu_result["passed"]
        meta["hallu_check"] = hallu_result

    # Примерная стоимость (DeepSeek: ~$0.5/M input tokens, ~$2/M output tokens)
    meta["cost_usd"] = (
        meta["prompt_tokens"] * 0.5e-6
        + meta["completion_tokens"] * 2.0e-6
    )

    return analysis, meta


def _build_analysis_prompt(
    md: MarketData,
    news: list[NewsArticle],
) -> str:
    """Собрать промпт с данными для LLM."""
    price_range = ""
    if md.high and md.low:
        price_range = f"за день: {md.low}–{md.high}"

    news_block = "\n\n".join(
        f"📰 «{a.title}»\n{a.snippet}"
        for a in news
    )

    return f"""Проанализируй акцию {md.company_name} (тикер {md.ticker}).

=== РЫНОЧНЫЕ ДАННЫЕ ===
Текущая цена: {md.price} ₽ {price_range}
Изменение за день: {md.change_percent:+.2f}%
Капитализация: {md.market_cap / 1e9:.1f} млрд ₽
Объём торгов сегодня: {md.volume_today:,} шт ({md.volume_today * md.price / 1e6:.1f} млн ₽)
Бид: {md.bid if md.bid else 'N/A'} | Оффер: {md.offer if md.offer else 'N/A'}
Спред: {md.spread if md.spread else 'N/A'}

=== НОВОСТИ ПО БУМАГЕ ===
{news_block}

На основе этих данных дай структурированную оценку акции.
"""


def _check_hallucinations(
    analysis: StockAnalysis,
    market_data: MarketData,
    news: list[NewsArticle],
) -> dict:
    """
    LLM-as-judge: проверка ответа на галлюцинации.

    Проверяет:
    1. Совпадают ли цены и капитализация с реальными MOEX-данными
    2. Факты из reasoning/key_factors присутствуют в MOEX-данных ИЛИ RAG-новостях
    3. Корректно ли пересчитаны единицы
    """
    errors = []

    # Проверка 1: цена
    expected_price = round(market_data.price, 2)
    if abs(analysis.current_price - expected_price) > 0.05 * expected_price:
        errors.append(
            f"Цена в ответе ({analysis.current_price}) "
            f"отличается от MOEX ({expected_price}) >5%"
        )

    # Проверка 2: капитализация (с точностью ±10% из-за округлений)
    expected_cap_bln = market_data.market_cap / 1e9
    if abs(analysis.market_cap_bln_rub - expected_cap_bln) > 0.1 * expected_cap_bln:
        errors.append(
            f"Капитализация в ответе ({analysis.market_cap_bln_rub:.1f}B) "
            f"отличается от MOEX ({expected_cap_bln:.1f}B) >10%"
        )

    # Проверка 3: объём
    expected_vol_mln = market_data.volume_today * market_data.price / 1e6
    if expected_vol_mln > 0:
        if abs(analysis.daily_volume_mln_rub - expected_vol_mln) > 0.2 * expected_vol_mln:
            errors.append(
                f"Объём в ответе ({analysis.daily_volume_mln_rub:.1f}M) "
                f"отличается от расчётного ({expected_vol_mln:.1f}M) >20%"
            )

    # Проверка 4: ghost-факты — LLM-as-judge с cross-check по новостям
    news_block = "\n---\n".join(
        f"НОВОСТЬ: {a.title}\nТЕКСТ: {a.snippet}" for a in news
    )

    judge_prompt = f"""Ты проверяешь анализ акции на галлюцинации (выдуманные неверные цифры или вымышленные события).

ИСХОДНЫЕ ДАННЫЕ (ground truth):
- Цена акции: {market_data.price} ₽
- Капитализация: {market_data.market_cap / 1e9:.1f} млрд ₽
- Изменение за день: {market_data.change_percent:+.2f}%
- Объём: {market_data.volume_today:,} шт

ИСХОДНЫЕ НОВОСТИ (только эти считаются реальными):
{news_block}

ОТВЕТ АНАЛИТИКА ПРОВЕРЯЕМ:
- bull факторы: {analysis.key_factors_bull}
- bear факторы: {analysis.key_factors_bear}
- Обоснование: {analysis.reasoning}

ВАЖНО: аналитические выводы и оценки (потенциал роста, уровень риска, перспективы) — это мнение, НЕ галлюцинации. Проверяй ТОЛЬКО конкретные цифры и упоминания событий.

Проверь: каждый ли утверждённый факт подтверждается MOEX-данными ИЛИ исходными новостями.

Ответь строго в формате JSON:
{{
  "verdict": "OK" или "HALLU",
  "hallu_facts": [список ТОЛЬКО неверных цифр или вымышленных событий],
  "reasoning": "краткое объяснение"
}}
"""

    judge_content, judge_usage = llm_complete(
        system_prompt="Ты строгий проверяющий фактов. Отвечай только JSON.",
        user_prompt=judge_prompt,
        temperature=0,
        max_retries=1,
    )

    # Парсим структурированный ответ судьи
    import re as _re
    import json as _json
    _m = _re.search(r'```(?:json)?\s*([\s\S]*?)```', judge_content)
    _clean = _m.group(1) if _m else judge_content
    try:
        _start = _clean.find('{')
        _end = _clean.rfind('}')
        if _start != -1 and _end != -1:
            _clean = _clean[_start:_end+1]
        judge_result = _json.loads(_clean)
        verdict = judge_result.get("verdict", "HALLU")
        if verdict == "HALLU":
            hallu_facts = judge_result.get("hallu_facts", [])
            errors.append(
                f"LLM-as-judge: галлюцинации — {'; '.join(hallu_facts)}"
            )
        judge_reasoning = judge_result.get("reasoning", "")
    except Exception:
        # Fallback: если парсинг не удался
        if "OK" not in judge_content.upper():
            errors.append(f"LLM-as-judge: не удалось распарсить вердикт — {judge_content[:200]}")
        judge_reasoning = judge_content[:200]

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "judge_verdict": judge_reasoning,
    }


def save_result(analysis: StockAnalysis, meta: dict, output_dir: str = "./output"):
    """Сохранить результат анализа в JSON."""
    import datetime as _dt
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if isinstance(obj, (_dt.date, _dt.datetime)):
            return obj.isoformat()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    filename = f"analysis_{analysis.ticker}_{analysis.analysis_date}.json"
    with open(path / filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "analysis": analysis.model_dump(),
                "meta": {k: v for k, v in meta.items() if k != "errors" or v},
            },
            f, ensure_ascii=False, indent=2, default=_serialize,
        )
    logger.info("Saved to %s", path / filename)
    return path / filename


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Инициализация
    corpus = NewsCorpus()
    corpus.load("./input/news")

    tickers = ["MAGN", "RAGR", "ALRS", "GMKN", "SVCB"]
    pipeline_results = []

    for ticker in tickers:
        print(f"Анализ: {ticker}")

        analysis, meta = run_analysis(ticker, corpus)

        if analysis:
            print(f"  Вердикт: {analysis.growth_outlook.upper()}")
            print(f"  Потенциал: {analysis.growth_potential_percent:+.1f}%")
            print(f"  Риск: {analysis.risk_level}")
            print(f"  Sentiment: {analysis.news_sentiment:+.2f}")
            print(f"  Токенов: {meta['total_tokens']}")
            print(f"  Hallu check: {'+' if analysis.hallu_check_passed else '❌'}")
            if meta.get("hallu_check", {}).get("errors"):
                for e in meta["hallu_check"]["errors"]:
                    print(f"    !  {e}")
            save_result(analysis, meta)

            pipeline_results.append({
                "ticker": analysis.ticker,
                "company_name": analysis.company_name,
                "analysis_date": str(analysis.analysis_date),
                "verdict": analysis.growth_outlook,
                "growth_potential_percent": analysis.growth_potential_percent,
                "risk_level": analysis.risk_level,
                "news_sentiment": analysis.news_sentiment,
                "hallu_check_passed": analysis.hallu_check_passed,
                "total_tokens": meta.get("total_tokens", 0),
                "cost_usd": round(meta.get("cost_usd", 0), 6),
            })
        else:
            print(f"  - Ошибка: {meta.get('errors')}")
            pipeline_results.append({
                "ticker": ticker,
                "company_name": "",
                "analysis_date": "",
                "verdict": "error",
                "growth_potential_percent": 0,
                "risk_level": "",
                "news_sentiment": 0,
                "hallu_check_passed": False,
                "total_tokens": meta.get("total_tokens", 0),
                "cost_usd": round(meta.get("cost_usd", 0), 6),
            })

    # Сохраняем сводку
    import datetime as _dt
    pipeline_summary = {
        "generated_at": str(_dt.datetime.now()),
        "total_analyses": len(tickers),
        "results": pipeline_results,
        "total_cost_usd": round(sum(r["cost_usd"] for r in pipeline_results), 6),
        "total_tokens": sum(r["total_tokens"] for r in pipeline_results),
        "hallu_passed_count": sum(1 for r in pipeline_results if r["hallu_check_passed"]),
    }
    out_path = Path("./output")
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "pipeline_results.json", "w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, ensure_ascii=False, indent=2)
    print(f"\nСводка сохранена в output/pipeline_results.json")
