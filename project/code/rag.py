"""
RAG-модуль: загрузка новостей, embeddings, поиск релевантных.

Используем sentence-transformers для мультиязычных эмбеддингов.
Храним всё в памяти — для 15 новостей по 5 бумагам этого достаточно.
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.linalg import norm

from schema import NewsArticle

logger = logging.getLogger(__name__)

# Эмбеддинги: lazy import — sentence-transformers грузится только при вызове
# Отключается через SENTENCE_TRANSFORMERS_DISABLE=1
EMBEDDING_AVAILABLE = True  # будет переключено на False при ошибке импорта
_model = None
_model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

def _get_embedder():
    """Lazy-загрузка sentence-transformers."""
    global _model, EMBEDDING_AVAILABLE

    if _model is not None:
        return _model

    if os.environ.get("SENTENCE_TRANSFORMERS_DISABLE"):
        EMBEDDING_AVAILABLE = False
        raise RuntimeError("Embeddings disabled by SENTENCE_TRANSFORMERS_DISABLE")

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", _model_name)
        _model = SentenceTransformer(_model_name)
        EMBEDDING_AVAILABLE = True
        return _model
    except Exception as e:
        logger.warning("sentence-transformers failed to load: %s. Using keyword fallback.", e)
        EMBEDDING_AVAILABLE = False
        raise RuntimeError("Embeddings not available")


def _use_keyword() -> bool:
    """Проверить, используем ли keyword fallback."""
    return os.environ.get("SENTENCE_TRANSFORMERS_DISABLE") or not EMBEDDING_AVAILABLE


class NewsCorpus:
    """
    Корпус новостей с поиском по косинусной близости.

    Пример:
        corpus = NewsCorpus()
        corpus.load("./input/news")
        results = corpus.search("металлургия", top_k=3)
    """

    def __init__(self):
        self.articles: list[NewsArticle] = []
        self._embeddings: Optional[np.ndarray] = None

    def load(self, news_dir: str = "./input/news"):
        """Загрузить все .json новости из папки."""
        path = Path(news_dir)
        if not path.exists():
            logger.warning("News directory not found: %s", news_dir)
            return

        self.articles = []
        for fpath in sorted(path.glob("*.json")):
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
                self.articles.append(NewsArticle(**data))

        logger.info("Loaded %d news articles from %s", len(self.articles), news_dir)
        self._compute_embeddings()

    def add(self, article: NewsArticle):
        """Добавить одну новость."""
        self.articles.append(article)
        self._embeddings = None  # Сбросить кеш

    def _compute_embeddings(self):
        """Вычислить эмбеддинги для всех статей."""
        if not self.articles:
            self._embeddings = None
            return

        if _use_keyword():
            self._embeddings = None
            return

        try:
            embedder = _get_embedder()
            texts = [f"{a.title}. {a.snippet}" for a in self.articles]
            self._embeddings = embedder.encode(texts, show_progress_bar=False)
        except Exception:
            self._embeddings = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[NewsArticle, float]]:
        """
        Поиск новостей по запросу.

        Returns:
            Список (статья, score) от наиболее релевантной
        """
        if not self.articles:
            return []

        # Keyword fallback (если эмбеддинги отключены или не загрузились)
        if _use_keyword() or self._embeddings is None:
            return self._keyword_search(query, top_k)

        # Embedding search
        embedder = _get_embedder()
        q_emb = embedder.encode([query], show_progress_bar=False)[0]

        scores = []
        for i, art_emb in enumerate(self._embeddings):
            score = float(np.dot(q_emb, art_emb) / (norm(q_emb) * norm(art_emb) + 1e-10))
            if score >= min_score:
                scores.append((i, score))

        scores.sort(key=lambda x: -x[1])
        return [(self.articles[i], s) for i, s in scores[:top_k]]

    def search_by_ticker(self, ticker: str, top_k: int = 3) -> list[tuple[NewsArticle, float]]:
        """Поиск новостей, релевантных конкретному тикеру."""
        # Сначала пробуем embedding-поиск по названию компании
        query = self._ticker_to_query(ticker)
        return self.search(query, top_k=top_k)

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[NewsArticle, float]]:
        """Примитивный keyword fallback."""
        terms = query.lower().split()
        scored = []
        for article in self.articles:
            text = f"{article.title} {article.snippet}".lower()
            score = sum(1 for t in terms if t in text) / max(len(terms), 1)
            scored.append((article, score))
        scored.sort(key=lambda x: -x[1])
        return [(a, s) for a, s in scored[:top_k] if s > 0]

    @staticmethod
    def _ticker_to_query(ticker: str) -> str:
        """Преобразовать тикер в поисковый запрос."""
        ticker_map = {
            "MAGN": "ММК металлургия сталь",
            "RAGR": "Русагро агрохолдинг сельское хозяйство",
            "ALRS": "Алроса алмазы",
            "GMKN": "Норникель никель металлы",
            "SVCB": "Совкомбанк банк финансы",
            "GAZP": "Газпром газ нефть",
            "SBER": "Сбербанк банк финансы",
            "LKOH": "Лукойл нефть топливо",
            "ROSN": "Роснефть нефть газ",
            "VTBR": "ВТБ банк финансы",
            "TATN": "Татнефть нефть НПЗ",
            "NLMK": "НЛМК сталь металлургия",
            "SNGS": "Сургутнефтегаз нефть дивиденды",
            "PIKK": "ПИК девелопмент недвижимость",
            "MGNT": "Магнит ритейл торговля",
        }
        return ticker_map.get(ticker, ticker)
