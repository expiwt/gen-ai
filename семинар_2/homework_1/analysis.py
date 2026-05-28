"""
analysis.py — расширенный анализ качества данных для ДПО-заявок.

На выходе:
 - ages.png — гистограмма возрастов
 - experience_by_speciality.png — violin(strip) стаж × специальность
 - cities.png — гистограмма по городам
 - specialities.png — гистограмма по специальностям
 - courses.png — гистограмма по курсам
 - cross_table_city_speciality.csv — кросс-таблица
 - report.md — текстовая сводка
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path.home() / "gen_ai" / "applications.csv"
OUTPUT_DIR = Path.home() / "gen_ai"


def load():
    df = pd.read_csv(CSV_PATH)
    print(f"Загружено: {len(df)} заявок из {CSV_PATH.name}")
    return df


def plot_hist_ages(df, out):
    plt.figure(figsize=(8, 4))
    plt.hist(df["age"], bins=12, color="#4A90D9", edgecolor="white")
    plt.xlabel("Возраст")
    plt.ylabel("Число заявок")
    plt.title(f"Распределение возраста ({len(df)} заявок)")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"   OK {out}")


def plot_bar(series, title, out, color, limit_pct=None):
    counts = series.value_counts()
    total = len(series)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(counts)), counts.values, color=color, edgecolor="white")
    plt.xticks(range(len(counts)), counts.index, rotation=30, ha="right")
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{count} ({pct:.0f}%)", ha="center", fontsize=9)
    if limit_pct:
        lim = total * limit_pct / 100
        plt.axhline(y=lim, color="red", linestyle="--", alpha=0.5,
                    label=f"порог {limit_pct:.0f}% ({lim:.0f})")
        plt.legend()
    plt.title(title)
    plt.ylabel("Число заявок")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"   OK {out}")
    return counts


def plot_violin(df, col_x, col_y, title, out):
    if col_y not in df.columns or col_x not in df.columns:
        return
    plt.figure(figsize=(12, 5))
    try:
        import seaborn as sns
        sns.violinplot(data=df, x=col_x, y=col_y, inner="quartile", density_norm="width")
        sns.stripplot(data=df, x=col_x, y=col_y, color="black", alpha=0.3, size=4)
    except ImportError:
        groups = df.groupby(col_x)[col_y].apply(list)
        positions = range(1, len(groups) + 1)
        plt.boxplot([list(v) for v in groups.values], positions=list(positions), vert=True)
        plt.xticks(list(positions), list(groups.index), rotation=30, ha="right")
    plt.ylabel(col_y)
    plt.xlabel(col_x)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"   OK {out}")


def write_report(df, out):
    n = len(df)
    lines = [f"# Отчёт по {n} заявкам на ДПО\n"]

    cities = df["city"].value_counts()
    top_city_pct = cities.iloc[0] / n * 100
    lines.append("## Города\n")
    lines.append(f"- Уникальных: {len(cities)} из 12 разрешённых")
    lines.append(f"- Топ-1: {cities.index[0]} — {cities.iloc[0]} ({top_city_pct:.0f}%)")
    lines.append(f"- Порог 40%: {'FAIL' if top_city_pct > 40 else 'OK'}")
    lines.append("")

    specs = df["speciality"].value_counts()
    top_spec_pct = specs.iloc[0] / n * 100
    lines.append("## Специальности\n")
    lines.append(f"- Уникальных: {len(specs)} из 12 разрешённых")
    lines.append(f"- Топ-1: {specs.index[0]} — {specs.iloc[0]} ({top_spec_pct:.0f}%)")
    lines.append(f"- Порог 35%: {'FAIL' if top_spec_pct > 35 else 'OK'}")
    lines.append("")

    courses = df["desired_course"].value_counts()
    lines.append("## Желаемые курсы\n")
    for course, count in courses.items():
        lines.append(f"- {course}: {count}")
    lines.append("")

    names = df["full_name"].value_counts()
    dupes = names[names > 1]
    lines.append("## Имена\n")
    lines.append(f"- Уникальных имён: {len(names)} из {n} ({len(names)/n*100:.0f}%)")
    if len(dupes):
        lines.append("- Повторы (топ-5):")
        for name, count in dupes.head(5).items():
            lines.append(f"  - {name} — {count} раз(а)")
    lines.append("")

    # Кросс-таблица
    ct = pd.crosstab(df["city"], df["speciality"])
    lines.append("## Кросс-таблица: город x специальность\n")
    lines.append("```")
    lines.append(ct.to_string())
    lines.append("```")

    lines.append("\n### Комментарий: 2-3 нереалистичные комбинации\n")
    ct_long = ct.melt(ignore_index=False).reset_index()
    ct_long.columns = ["city", "speciality", "count"]
    pairs = ct_long[ct_long["count"] > 0][["city", "speciality"]]

    lines.append(
        "1. **Врач-терапевт -> Data Science** — врач, идущий на курс по машинному "
        "обучению, возможен (медицинская информатика), но скорее редкость. "
        "Правдоподобнее был бы «Клиническая медицина (обновление)».\n"
    )
    lines.append(
        "2. **Программист -> AutoCAD и BIM** — строительный софт для разработчика "
        "возможен (если он в CivilTech), но статистически маловероятен.\n"
    )
    lines.append(
        "3. **Воспитатель -> Клиническая медицина** — педагог-дошкольник, "
        "записывающийся на медкурс, выглядит натянуто.\n"
    )
    lines.append("")

    # Стаж
    lines.append("## Стаж по специальностям\n")
    med = df.groupby("speciality")["years_of_experience"].agg(["min", "median", "max"])
    lines.append("```")
    lines.append(med.to_string())
    lines.append("```")
    lines.append("")

    # Возраст и год выпуска
    lines.append("## Возраст и год выпуска\n")
    lines.append(f"- Средний возраст: {df['age'].mean():.1f} лет")
    lines.append(f"- Средний стаж: {df['years_of_experience'].mean():.1f} лет")
    lines.append(f"- Средний год выпуска: {df['graduation_year'].mean():.0f}")

    conflicts = 0
    for _, row in df.iterrows():
        age_at_grad = row["graduation_year"] - (2025 - row["age"])
        if age_at_grad < 17 or age_at_grad > 40:
            conflicts += 1
    lines.append(f"\n- Конфликтов age/graduation_year: {conflicts} "
                 f"{'OK' if conflicts == 0 else 'FAIL'}")

    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print(f"   OK {out}")


def main():
    df = load()

    plot_hist_ages(df, str(OUTPUT_DIR / "ages.png"))
    plot_bar(df["city"], "Распределение по городам",
             str(OUTPUT_DIR / "cities.png"), "#7AB66E", limit_pct=40)
    plot_bar(df["speciality"], "Распределение по специальностям",
             str(OUTPUT_DIR / "specialities.png"), "#D97A4A", limit_pct=35)
    plot_bar(df["desired_course"], "Распределение по курсам",
             str(OUTPUT_DIR / "courses.png"), "#5B9BD5")
    plot_violin(df, "speciality", "years_of_experience",
                "Стаж по специальностям",
                str(OUTPUT_DIR / "experience_by_speciality.png"))

    ct = pd.crosstab(df["city"], df["speciality"], margins=True, margins_name="Всего")
    ct.to_csv(OUTPUT_DIR / "cross_table_city_speciality.csv")
    print(f"   OK cross_table_city_speciality.csv")

    write_report(df, str(OUTPUT_DIR / "report.md"))
    print(f"\nАнализ завершён. Файлы в {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
