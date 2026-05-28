"""
schema.py — Pydantic-модели Address и Application для заявок на ДПО.

Специальности (минимум 8):
    Врач-терапевт, Инженер-строитель, Учитель математики, Бухгалтер,
    HR-менеджер, Программист, Менеджер по продажам, Экономист,
    Юрист, Дизайнер, Маркетолог, Воспитатель

Курсы (минимум 6):
    Управление проектами, Цифровая трансформация бизнеса,
    Data Science и машинное обучение, Управление персоналом,
    Финансовый менеджмент, Педагогика и психология,
    Клиническая медицина (обновление), AutoCAD и BIM-технологии
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


CITIES = {
    "Москва", "Санкт-Петербург", "Казань", "Екатеринбург",
    "Новосибирск", "Нижний Новгород", "Ростов-на-Дону",
    "Самара", "Красноярск", "Воронеж", "Пермь", "Волгоград",
}


class Address(BaseModel):
    """Вложенная модель адреса."""
    city: str
    district: str = Field(min_length=2, max_length=40)

    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Город «{v}» не из утверждённого списка")
        return v


class Application(BaseModel):
    """Заявка на курс повышения квалификации (ДПО)."""
    full_name: str = Field(description="ФИО полностью (русские имя, фамилия, отчество)")
    age: int = Field(ge=22, le=65, description="Возраст от 22 до 65")
    address: Address = Field(description="Адрес (вложенная модель: city + district)")
    speciality: Literal[
        "Врач-терапевт",
        "Инженер-строитель",
        "Учитель математики",
        "Бухгалтер",
        "HR-менеджер",
        "Программист",
        "Менеджер по продажам",
        "Экономист",
        "Юрист",
        "Дизайнер",
        "Маркетолог",
        "Воспитатель",
    ] = Field(description="Текущая специальность заявителя")
    desired_course: Literal[
        "Управление проектами",
        "Цифровая трансформация бизнеса",
        "Data Science и машинное обучение",
        "Управление персоналом",
        "Финансовый менеджмент",
        "Педагогика и психология",
        "Клиническая медицина (обновление)",
        "AutoCAD и BIM-технологии",
    ] = Field(description="Желаемый курс повышения квалификации")
    years_of_experience: int = Field(ge=0, le=40, description="Лет опыта от 0 до 40")
    graduation_year: int = Field(ge=1980, le=2024, description="Год окончания вуза")

    @field_validator("graduation_year")
    @classmethod
    def check_graduation_age_consistency(cls, v: int, info) -> int:
        """
        Проверяет, что год выпуска не противоречит возрасту.

        Если graduation_year < (2025 - age) + 17 → человек «выпустился»
        до своего рождения или в детстве — ошибка.
        Если graduation_year > (2025 - age) + 40 → слишком поздно (>40 лет при выпуске).
        """
        current_year = 2025
        birth_year = current_year - info.data.get("age", 30)
        age_at_graduation = v - birth_year

        if age_at_graduation < 17:
            raise ValueError(
                f"Год окончания {v} слишком ранний для возраста "
                f"{info.data.get('age')}: возраст при выпуске {age_at_graduation} лет "
                f"(минимум 17)"
            )
        if age_at_graduation > 40:
            raise ValueError(
                f"Год окончания {v} слишком поздний для возраста "
                f"{info.data.get('age')}: возраст при выпуске {age_at_graduation} лет "
                f"(максимум 40)"
            )
        return v

    @property
    def city(self) -> str:
        """Удобный shortcut: app.city вместо app.address.city."""
        return self.address.city
