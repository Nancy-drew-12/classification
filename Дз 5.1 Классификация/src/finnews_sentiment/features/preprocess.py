from typing import List

import html
import re
import string

import pandas as pd
import nltk

# Убедитесь, что необходимые ресурсы NLTK скачаны:
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("wordnet")
# nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

TICKER_PATTERN = re.compile(r"\$[A-Za-z]{1,6}\b")  # тикеры в твитах: $AAPL, $ESI


def clean_text(text: str) -> str:
    """Лёгкая очистка для тональности.

    - html-сущности (&amp; &quot; &#39;) -> обычные символы
    - нижний регистр
    - удаление html-тегов и ссылок
    - лишние пробелы

    Цифры, проценты, $ и стоп-слова (not, no, down) НЕ удаляем:
    для финансовой тональности они несут смысл.
    """
    if not isinstance(text, str):
        return ""
    txt = html.unescape(text)
    txt = txt.lower()
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"http\S+|www\.\S+", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def preprocess_text(text: str) -> str:
    """Агрессивная предобработка из шаблона: лемматизация + удаление стоп-слов,
    цифр и пунктуации. Используется как отдельный эксперимент (день 4)."""
    txt = clean_text(text)
    txt = txt.translate(str.maketrans("", "", string.digits))
    txt = txt.translate(str.maketrans("", "", string.punctuation))

    tokens = nltk.word_tokenize(txt)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def clean_news_data(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Применяет clean_text ко всему DataFrame, создаёт колонку `text_clean`."""
    df = df.copy()
    df["text_clean"] = df[text_col].astype(str).map(clean_text)
    return df


def add_basic_features(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    """Простые числовые признаки на основе текста."""
    df = df.copy()
    df["word_count_clean"] = df[text_col].astype(str).str.split().str.len()
    df["char_count"] = df[text_col].astype(str).str.len()
    # в README count('$'), но '$' в регулярке - конец строки, поэтому экранируем
    df["dollar_count"] = df[text_col].astype(str).str.count(r"\$")
    df["ticker_count"] = df["text"].astype(str).apply(lambda s: len(TICKER_PATTERN.findall(s)))
    df["number_count"] = df["text"].astype(str).str.findall(r"\d+(?:\.\d+)?").apply(len)
    return df


def prepare_dataset(df: pd.DataFrame, text_col: str = "text", label_col: str = "sentiment") -> pd.DataFrame:
    """Полная подготовка: пропуски, конфликтные дубли, очистка, пустые тексты, метки."""
    df = df.dropna(subset=[text_col, label_col]).copy()

    # тексты, у которых в датасете разные метки, - шум разметки, убираем целиком
    n_labels = df.groupby(text_col)[label_col].transform("nunique")
    df = df[n_labels == 1]
    df = df.drop_duplicates(subset=[text_col])

    df = clean_news_data(df, text_col=text_col)
    df = df[df["text_clean"].str.len() > 0]
    df = df.drop_duplicates(subset=["text_clean"])

    df["label"] = df[label_col].map(LABEL_MAP)
    df = add_basic_features(df)
    return df.reset_index(drop=True)