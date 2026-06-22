"""
Клиент для MOEX ISS API.

Без токена, без аутентификации, работает из коробки.

Документация: https://iss.moex.com/iss/reference/
"""
from datetime import datetime
from typing import Optional

import requests

from schema import MarketData


MOEX_BASE = "https://iss.moex.com/iss"
BOARD = "TQBR"  # основной режим торгов (акции)


def _extract_current_marketcap(ticker: str) -> float:
    """Получить текущую капитализацию как прокси для исторических данных."""
    url = (
        f"{MOEX_BASE}/engines/stock/markets/shares/boards/{BOARD}"
        f"/securities/{ticker}.json?iss.meta=off&iss.only=marketdata"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        md_cols = data.get("marketdata", {}).get("columns", [])
        md_rows = data.get("marketdata", {}).get("data", [])
        if md_rows:
            md = dict(zip(md_cols, md_rows[0]))
            cap = md.get("ISSUECAPITALIZATION")
            if cap:
                return float(cap)
    except Exception:
        pass
    return 0.0


def fetch_marketdata(ticker: str) -> Optional[MarketData]:
    """
    Получить текущие биржевые данные для тикера.

    Пример:
        data = fetch_marketdata("MAGN")
        print(data.price, data.market_cap)
    """
    url = (
        f"{MOEX_BASE}/engines/stock/markets/shares/boards/{BOARD}"
        f"/securities/{ticker}.json?iss.meta=off&iss.only=securities,marketdata"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception:
        return None
    data = resp.json()

    # Парсим securities (название компании)
    sec_cols = data.get("securities", {}).get("columns", [])
    sec_rows = data.get("securities", {}).get("data", [])
    if not sec_rows:
        return None
    sec_info = dict(zip(sec_cols, sec_rows[0]))

    # Парсим marketdata (цены, капитализация)
    md_cols = data.get("marketdata", {}).get("columns", [])
    md_rows = data.get("marketdata", {}).get("data", [])
    if not md_rows:
        return None
    md = dict(zip(md_cols, md_rows[0]))

    def _f(key: str) -> Optional[float]:
        val = md.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _i(key: str) -> int:
        val = md.get(key)
        if val is None:
            return 0
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    return MarketData(
        ticker=ticker,
        company_name=sec_info.get("SHORTNAME", ticker),
        price=_f("LAST") or _f("LCURRENTPRICE") or 0.0,
        open_price=_f("OPEN"),
        high=_f("HIGH"),
        low=_f("LOW"),
        change_percent=_f("CHANGE") or 0.0,
        market_cap=float(_f("ISSUECAPITALIZATION") or 0),
        volume_today=_i("VOLTODAY"),
        num_trades=_i("NUMTRADES"),
        bid=_f("BID"),
        offer=_f("OFFER"),
        spread=_f("SPREAD"),
        fetched_at=datetime.now(),
    )


def fetch_marketdata_by_date(ticker: str, trade_date: str) -> Optional[MarketData]:
    """
    Получить биржевые данные на конкретную торговую дату.

    Использует history endpoint MOEX.
    Капитализация берётся из текущих данных как прокси.

    Пример:
        data = fetch_marketdata_by_date("MAGN", "2026-06-10")
    """
    url = (
        f"{MOEX_BASE}/history/engines/stock/markets/shares/boards/{BOARD}"
        f"/securities/{ticker}.json"
        f"?from={trade_date}&till={trade_date}&iss.meta=off&iss.only=history"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception:
        return None
    data = resp.json()

    rows = data.get("history", {}).get("data", [])
    if not rows:
        return None

    cols = data["history"]["columns"]
    row = dict(zip(cols, rows[0]))

    def _f(key: str) -> Optional[float]:
        val = row.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _i(key: str) -> int:
        val = row.get(key)
        if val is None:
            return 0
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    close = _f("CLOSE") or _f("LEGALCLOSEPRICE") or 0.0
    open_p = _f("OPEN")
    high = _f("HIGH")
    low = _f("LOW")
    volume = _i("VOLUME")
    num_trades = _i("NUMTRADES")

    # Изменение за день: (close - open) / open * 100
    change_pct = 0.0
    if open_p and open_p > 0:
        change_pct = round((close - open_p) / open_p * 100, 2)

    # Капитализация как прокси (из текущих данных)
    market_cap = _extract_current_marketcap(ticker)

    return MarketData(
        ticker=ticker,
        company_name=str(row.get("SHORTNAME", ticker)),
        price=close,
        open_price=open_p,
        high=high,
        low=low,
        change_percent=change_pct,
        market_cap=market_cap,
        volume_today=volume,
        num_trades=num_trades,
        bid=None,
        offer=None,
        spread=None,
        fetched_at=datetime.now(),
    )
