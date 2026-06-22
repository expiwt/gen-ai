# Stock Analyzer — финальный проект

Автоматический анализ российских акций на основе данных MOEX + новостного RAG.

**Трек:** B (прикладной)  
**Техники курса:** RAG, Agent, LLM-as-Judge, Structured Output, Pydantic-валидация  
**Данные:** MOEX ISS API (открытый) + 90 новостей по 9+ бумагам

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

# 4. Eval на 50+ тестах
python eval.py

# 5. Графики
python code/charts.py
```

Результаты — в `output/`, `output/eval/`, графики — в `output/eval/charts/`.

---

## Структура

```
project/
├── run.sh                    ← скрипт одной команды
├── README.md
├── requirements.txt
├── .env.example              ← без токена
├── eval.py                   ← 56 тестов, метрики, LLM-as-judge
├── отчёт.md                  ← отчёт 5 разделов
├── code/
│   ├── schema.py             ← Pydantic + field_validator'ы
│   ├── moex_client.py        ← MOEX ISS API (агентский инструмент)
│   ├── rag.py                ← RAG (embeddings / keyword fallback)
│   ├── llm_client.py         ← JSON mode + max_retries (response_model)
│   ├── pipeline.py           ← агент: MOEX → RAG → LLM → hallu check
│   └── charts.py             ← matplotlib-графики по eval
├── input/
│   └── news/                 ← 90 новостей по 9+ бумагам (JSON)
├── output/                   ← артефакты прогона
│   ├── analysis_*.json       ← анализы
│   ├── pipeline_results.json ← сводка
│   └── eval/                 ← eval_results.csv + .json + summary + charts/
└── venv/
```

---

## Техники курса (5 из ≥4)

| # | Техника | Где |
|---|---------|-----|
| 1 | **RAG** | `rag.py` — sentence-transformers → эмбеддинги → retrieval. Fallback: keyword |
| 2 | **Agent с инструментами** | `pipeline.py` — MOEX API (+ RAG + LLM) как инструменты |
| 3 | **LLM-as-judge** | `pipeline.py.check_hallucinations()` + `eval.py.judge_eval()` — сверка и с MOEX, и с RAG-новостями |
| 4 | **Structured output** | `schema.py` response_model + `llm_client.py` max_retries=3 |
| 5 | **field_validator** | `schema.py` — 4 бизнес-инварианта + model_validator |

## Бумаги

| Тикер | Компания | Сектор |
|-------|----------|--------|
| MAGN | ММК | Чёрная металлургия |
| RAGR | РусАгро | Агросектор |
| ALRS | Алроса | Добыча алмазов |
| GMKN | Норникель | Цветные металлы |
| SVCB | Совкомбанк | Банкинг |
| GAZP | Газпром | Нефть и газ |
| SBER | Сбербанк | Банкинг |
| LKOH | Лукойл | Нефть и газ |
| ROSN | Роснефть | Нефть и газ |

## Результаты eval

| Метрика | Значение |
|---------|----------|
| Тестов | 56 (4 тикера × 14 еженедельных дат) |
| Pass rate | 35/56 (62%) |
| Avg correctness | 3.2 / 5.0 |
| Avg tokens / run | 1 327 |
| Общая стоимость | $0.064 |
| Hallu pass | 40/56 (71%) |
| Avg latency | 11.75 с |

Подробности — в `output/eval/eval_results.csv`, графиках `output/eval/charts/` и `отчёт.md`.
