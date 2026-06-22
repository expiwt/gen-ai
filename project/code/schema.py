"""
Схемы данных для Stock Analyzer.

- MarketData — сырые данные с MOEX
- NewsArticle — одна новость
- StockAnalysis — итоговый анализ (response_model)
"""

from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class MarketData(BaseModel):
    """Снапшот биржевых данных."""
    ticker: str = Field(description="Тикер на Мосбирже")
    company_name: str = Field(description="Краткое наименование")
    price: float = Field(gt=0, description="Текущая цена (LAST)")
    open_price: Optional[float] = Field(default=None, ge=0)
    high: Optional[float] = Field(default=None, ge=0)
    low: Optional[float] = Field(default=None, ge=0)
    change_percent: float = Field(description="Изменение за день в %")
    market_cap: float = Field(ge=0, description="Рыночная капитализация")
    volume_today: int = Field(ge=0, description="Объём торгов сегодня")
    num_trades: int = Field(ge=0, description="Количество сделок")
    bid: Optional[float] = Field(default=None, ge=0)
    offer: Optional[float] = Field(default=None, ge=0)
    spread: Optional[float] = Field(default=None, ge=0)
    fetched_at: datetime = Field(description="Время получения данных")


class NewsArticle(BaseModel):
    """Одна новость по бумаге."""
    title: str = Field(description="Заголовок")
    source: str = Field(default="SmartLab", description="Источник")
    date: Optional[str] = Field(default=None, description="Дата новости")
    snippet: str = Field(description="Текст/краткое содержание")
    relevance_score: Optional[float] = Field(default=None, ge=0, le=1)


class StockAnalysis(BaseModel):
    """
    Итоговый анализ акции.
    Используется как response_model для LLM.
    """
    ticker: str = Field(description="Тикер")
    company_name: str = Field(description="Название")
    analysis_date: date = Field(description="Дата анализа")

    # --- Рыночные метрики ---
    current_price: float = Field(ge=0, description="Текущая цена")
    price_change_1d_percent: float = Field(description="Изменение за день %")
    market_cap_bln_rub: float = Field(ge=0, description="Капитализация в млрд ₽")
    daily_volume_mln_rub: float = Field(ge=0, description="Дневной объём в млн ₽")

    # --- Новостной фон ---
    news_sentiment: float = Field(
        ge=-1.0, le=1.0,
        description="Тональность новостей: -1 негатив → +1 позитив"
    )
    top_news_themes: list[str] = Field(
        description="Ключевые темы новостей (до 3)",
        max_length=3,
    )

    # --- Оценки ---
    growth_potential_percent: float = Field(
        ge=-50, le=200,
        description="Потенциал роста в % (от -50 до +200)"
    )
    growth_outlook: str = Field(
        description="Вердикт: strong_buy / buy / hold / sell / strong_sell"
    )
    risk_level: str = Field(
        description="Уровень риска: низкий / средний / высокий"
    )
    key_factors_bull: list[str] = Field(
        description="Поддерживающие факторы (бычьи)"
    )
    key_factors_bear: list[str] = Field(
        description="Давящие факторы (медвежьи)"
    )

    # --- Мета ---
    reasoning: str = Field(description="Краткое обоснование вывода")
    hallu_check_passed: bool = Field(
        description="Прошла ли проверка галлюцинаций"
    )

    # ---- field_validators ----

    @field_validator("growth_outlook")
    @classmethod
    def valid_outlook(cls, v: str) -> str:
        allowed = {"strong_buy", "buy", "hold", "sell", "strong_sell"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"growth_outlook должен быть одним из {allowed}, получено '{v}'"
            )
        return v_lower

    @field_validator("risk_level")
    @classmethod
    def valid_risk(cls, v: str) -> str:
        allowed = {"низкий", "средний", "высокий"}
        if v not in allowed:
            raise ValueError(
                f"risk_level должен быть одним из {allowed}, получено '{v}'"
            )
        return v

    @field_validator("top_news_themes")
    @classmethod
    def non_empty_themes(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("top_news_themes не может быть пустым")
        return v

    @field_validator("analysis_date")
    @classmethod
    def not_future(cls, v: date) -> date:
        if v > date.today():
            raise ValueError(f"analysis_date ({v}) не может быть в будущем")
        return v

    @model_validator(mode="after")
    def consistent_outlook_and_potential(self):
        """growth_outlook и growth_potential_percent должны быть согласованы."""
        outlook = self.growth_outlook
        potential = self.growth_potential_percent
        if outlook == "strong_buy" and potential < 15:
            raise ValueError(
                f"strong_buy предполагает потенциал ≥15%, получено {potential}%"
            )
        if outlook == "strong_sell" and potential > -10:
            raise ValueError(
                f"strong_sell предполагает потенциал ≤-10%, получено {potential}%"
            )
        return self


class EvalResult(BaseModel):
    """Результат одного eval-прогона."""
    ticker: str
    analysis_date: str
    # Оценка правильности
    correctness_score: float = Field(ge=0, le=5, description="Оценка LLM-as-judge, 0–5")
    price_match: bool = Field(description="Совпала ли цена с реальной")
    no_ghost_facts: bool = Field(description="Нет выдуманных цифр/фактов")
    # Путь
    steps: int = Field(ge=1, description="Число шагов агента")
    tokens_used: int = Field(ge=0, description="Всего токенов")
    cost_usd: float = Field(ge=0, description="Стоимость прогона $")
    # Итог
    passed: bool = Field(description="Тест пройден")
    errors: list[str] = Field(default_factory=list, description="Что пошло не так")
