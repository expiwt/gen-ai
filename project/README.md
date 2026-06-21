# Stock Analyzer — финальный проект

Автоматический анализ российских акций на основе данных MOEX + новостного RAG.

**Трек:** B (прикладной)  
**Техники курса:** RAG, Agent, LLM-as-Judge, Structured Output, Pydantic-валидация  
**Данные:** MOEX ISS API (открытый) + 15 новостей SmartLab

---

## Одна команда запуска

```bash
./run.sh
```

Или вручную:

```bash
# 1. venv + зависимости
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. API-ключ
cp .env.example .env
# → отредактировать .env: вставить OPENAI_API_KEY (DeepSeek / OpenAI / любой)

# 3. Анализ 5 бумаг
python code/pipeline.py

# 4. Eval на 15 тестах
python eval.py
```

Результаты появятся в `output/` и `output/eval/`.

---

## Структура

```
project/
├── run.sh                    ← скрипт одной команды
├── README.md
├── requirements.txt
├── .env.example              ← без токена
├── eval.py                   ← 15 тестов, метрики, LLM-as-judge
├── отчёт.md                  ← отчёт 5 разделов
├── code/
│   ├── schema.py             ← Pydantic + field_validator'ы
│   ├── moex_client.py        ← MOEX ISS API (агентский инструмент)
│   ├── rag.py                ← RAG (embeddings / keyword fallback)
│   ├── llm_client.py         ← JSON mode + max_retries (response_model)
│   └── pipeline.py           ← агент: MOEX → RAG → LLM → hallu check
├── input/
│   └── news/                 ← 15 новостей по 5 бумагам (JSON)
├── output/                   ← артефакты прогона
│   ├── analysis_*.json       ← 5 анализов
│   ├── pipeline_results.json ← сводка
│   └── eval/                 ← eval_results.csv + .json + summary
└── venv/
```

---

## Техники курса (5 из ≥4)

| # | Техника | Где |
|---|---------|-----|
| 1 | **RAG** | `rag.py` — sentence-transformers → эмбеддинги → retrieval. Fallback: keyword |
| 2 | **Agent с инструментами** | `pipeline.py` — MOEX API (+ RAG + LLM) как инструменты |
| 3 | **LLM-as-judge** | `pipeline.py.check_hallucinations()` + `eval.py.judge_eval()` |
| 4 | **Structured output** | `schema.py` response_model + `llm_client.py` max_retries=3 |
| 5 | **field_validator** | `schema.py` — 4 бизнес-инварианта + model_validator |

## Бумаги

| Тикер | Компания | Сектор | Результат |
|-------|----------|--------|-----------|
| MAGN | ММК | Чёрная металлургия | BUY +10–12% |
| RAGR | РусАгро | Агросектор | BUY +10–12% |
| ALRS | Алроса | Добыча алмазов | BUY +5% |
| GMKN | Норникель | Цветные металлы | BUY +5% |
| SVCB | Совкомбанк | Банкинг | BUY/STRONG_BUY +15–20% |

## Результаты eval

| Метрика | Значение |
|---------|----------|
| Pass rate | 15/15 (100%) |
| Avg correctness | 4.33 / 5.0 |
| Avg tokens / run | 1 643 |
| Общая стоимость | $0.022 |
| Hallu pass | 15/15 (100%) |
| Сценарии | Реальные исторические даты (19.06 / 05.06 / 18.05) |

Подробности — в `output/eval/eval_results.csv` и `отчёт.md`.
