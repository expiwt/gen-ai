"""
generator.py — генерация 50 заявок на курсы ДПО.
"""

import os
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm_client import make_client
from schema import Application


OUTPUT_DIR = Path.home() / "gen_ai"

CITIES = [
    "Москва", "Санкт-Петербург", "Казань", "Екатеринбург",
    "Новосибирск", "Нижний Новгород", "Ростов-на-Дону",
    "Самара", "Красноярск", "Воронеж", "Пермь", "Волгоград",
]

SPECIALITIES = [
    "Врач-терапевт", "Инженер-строитель", "Учитель математики", "Бухгалтер",
    "HR-менеджер", "Программист", "Менеджер по продажам", "Экономист",
    "Юрист", "Дизайнер", "Маркетолог", "Воспитатель",
]


def build_prompt(city: str, seed_spec: str = None) -> str:
    prompt = (
        f"Сгенерируй одну заявку на курс повышения квалификации (ДПО).\n\n"
        f"Город заявителя: {city}.\n\n"
        f"Поля:\n"
        f"- full_name: ФИО (русское, полностью)\n"
        f"- age: от 22 до 65\n"
        f"- address: city + district\n"
        f"- speciality: строго из списка\n"
        f"- desired_course: строго из списка\n"
        f"- years_of_experience: 0-40\n"
        f"- graduation_year: 1980-2024\n\n"
        f"Специальности: Врач-терапевт, Инженер-строитель, Учитель математики, "
        f"Бухгалтер, HR-менеджер, Программист, Менеджер по продажам, "
        f"Экономист, Юрист, Дизайнер, Маркетолог, Воспитатель.\n\n"
        f"Курсы: Управление проектами, Цифровая трансформация бизнеса, "
        f"Data Science и машинное обучение, Управление персоналом, "
        f"Финансовый менеджмент, Педагогика и психология, "
        f"Клиническая медицина (обновление), AutoCAD и BIM-технологии.\n\n"
        f"ВАЖНО: используй ВСЕ 12 специальностей равномерно — не зацикливайся "
        f"на 2-3. Комбинируй специальности и курсы правдоподобно, но разнообразно. "
        f"ФИО — не повторяй одни и те же. District — реальный район указанного города."
    )
    if seed_spec:
        prompt += f"\n\nПопробуй для этой заявки специальность: {seed_spec}."
    return prompt


def main():
    print("Запуск генерации 50 заявок...")
    model = os.environ.get("LLM_MODEL", "deepseek-chat")
    print(f"   Модель: {model}")

    client = make_client()

    # Перемешиваем порядок городов случайно (не циклически)
    city_order = CITIES * 5  # 60 шт, хватит с запасом
    random.shuffle(city_order)
    city_order = city_order[:50]

    # Специальности: случайный seed для каждого вызова
    spec_seeds = SPECIALITIES * 5
    random.shuffle(spec_seeds)
    spec_seeds = spec_seeds[:50]

    applications = []

    for i in range(50):
        city = city_order[i]
        spec_seed = spec_seeds[i]
        prompt = build_prompt(city, seed_spec=spec_seed)

        for attempt in range(4):
            try:
                app = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=Application,
                    max_retries=3,
                    temperature=0.8 + attempt * 0.05,
                )
                applications.append(app)
                print(f"   [{i+1:2d}/50] {app.full_name[:25]:25s} | {app.city:15s} | {app.speciality:20s}")
                break
            except Exception as e:
                if attempt >= 3:
                    print(f"   [{i+1:2d}/50] Ошибка: {e}")

    print(f"\nСгенерировано: {len(applications)} заявок")

    # CSV
    rows = []
    for app in applications:
        rows.append({
            "full_name": app.full_name, "age": app.age,
            "city": app.city, "district": app.address.district,
            "speciality": app.speciality, "desired_course": app.desired_course,
            "years_of_experience": app.years_of_experience,
            "graduation_year": app.graduation_year,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "applications.csv", index=False)
    print("CSV: applications.csv")

    # Гистограммы
    for col, title, fname, color, lim in [
        ("city", "Распределение по городам", "cities.png", plt.cm.tab20c, 40),
        ("speciality", "Распределение по специальностям", "specialities.png", plt.cm.Set2, 35),
    ]:
        counts = Counter(getattr(app, col) if hasattr(app, col)
                         else app[col] if isinstance(app, dict)
                         else getattr(app, col)
                         for app in applications)

        if col == "city":
            counts = Counter(app.city for app in applications)
        elif col == "speciality":
            counts = Counter(app.speciality for app in applications)

        items = sorted(counts.keys(), key=lambda c: -counts[c])
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(items, [counts[c] for c in items], color=color(range(len(items))))
        for bar, c in zip(bars, [counts[c] for c in items]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(c), ha="center", fontsize=10)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels(items, rotation=45, ha="right")
        total_records = len(applications)
        limit_line = total_records * lim / 100
        ax.axhline(y=limit_line, color="red", linestyle="--", alpha=0.6,
                   label=f"{lim}% ({limit_line:.0f})")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / fname, dpi=150)
        plt.close(fig)
        print(f"PNG: {fname}")

    # Проверка
    city_counts = Counter(app.city for app in applications)
    spec_counts = Counter(app.speciality for app in applications)
    max_city_pct = max(city_counts.values()) / 50 * 100
    max_spec_pct = max(spec_counts.values()) / 50 * 100
    print(f"\nГорода: max {max_city_pct:.1f}% (<=40% {'OK' if max_city_pct <= 40 else 'FAIL'})")
    print(f"Спец-ти: max {max_spec_pct:.1f}% (<=35% {'OK' if max_spec_pct <= 35 else 'FAIL'})")
    print(f"Всего: {len(applications)}/50 OK")


if __name__ == "__main__":
    main()
