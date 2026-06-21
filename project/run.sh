#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo " Stock Analyzer"
echo ""

# 1. Проверить/создать venv
if [ ! -d venv ]; then
    echo "[1/4] Создаю виртуальное окружение..."
    python3 -m venv venv
fi

source venv/bin/activate

# 2. Установить зависимости (без sentence-transformers — подгрузится при необходимости)
echo "[2/4] Устанавливаю зависимости..."
pip install -q -r requirements.txt 2>&1 | tail -1

# 3. Проверить .env
if ! grep -q "sk-" .env 2>/dev/null; then
    echo "[!] Файл .env не настроен."
    echo "    Сделайте: cp .env.example .env"
    echo "    и вставьте OPENAI_API_KEY (DeepSeek / OpenAI)"
    exit 1
fi

# 4. Запустить пайплайн
echo "[3/4] Анализ 5 бумаг..."
python code/pipeline.py 2>&1

# 5. Запустить eval
echo ""
echo "[4/4] Eval на 15 тестах..."
python eval.py 2>&1 | grep -v "^INFO: HTTP Request"


echo "Результаты:  output/eval/eval_summary.json"
echo "Отчёт:       отчёт.md"
