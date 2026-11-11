from typing import List

import pandas as pd
import re
import string

import nltk

# Убедитесь, что необходимые ресурсы NLTK скачаны:
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

TICKER_PATTERN = re.compile(r"\b[A-Z]{2,6}\b")  # грубое приближение тикеров


def preprocess_text(text: str) -> str:
    """Базовая предобработка текста новости.

    Шаги:
    - приведение к нижнему регистру
    - удаление HTML-тегов (грубое)
    - замена тикеров специальным токеном
    - удаление спецсимволов и цифр
    - токенизация и лемматизация
    - удаление стоп-слов
    """
    if not isinstance(text, str):
        return ""

    # артефакты AG News: #39; quot; amp;
    txt = re.sub(r"\s?#39;", "'", text)
    txt = re.sub(r"\s?(quot|amp|lt|gt);", " ", txt)
    txt = txt.replace("\\", " ")

    txt = txt.lower()

    # простое удаление HTML тегов
    txt = re.sub(r"<.*?>", " ", txt)

    # удаление цифр и пунктуации
    txt = txt.translate(str.maketrans("", "", string.digits))
    txt = txt.translate(str.maketrans("", "", string.punctuation))

    tokens = nltk.word_tokenize(txt)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 2]

    return " ".join(tokens)


def clean_news_data(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Применяет preprocess_text ко всему DataFrame.

    Создаёт новую колонку `text_clean`.
    """
    df = df.copy()
    df["text_clean"] = df[text_col].astype(str).map(preprocess_text)
    return df


def add_basic_features(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    """Добавляет простые числовые признаки на основе текста.

    Примеры:
    - длина текста в словах
    - количество тикеров
    - количество чисел в исходном тексте
    """
    df = df.copy()
    df["text_len_words"] = df[text_col].astype(str).str.split().apply(len)
    df["ticker_count"] = df["text"].astype(str).apply(lambda s: len(TICKER_PATTERN.findall(s)))

    # количество чисел в исходном тексте
    df["number_count"] = df["text"].astype(str).str.findall(r"\d+(?:\.\d+)?").apply(len)

    return df

def prepare_dataset(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Очистка + удаление пустых и дублей + метки классов."""
    df = clean_news_data(df, text_col=text_col)
    df = df[df["text_clean"].str.split().str.len() >= 3]
    df = df.drop_duplicates(subset="text_clean")
    df["label"] = df[label_col].astype(int)
    df["label_name"] = df["label"].map(LABEL_NAMES)
    return df.reset_index(drop=True)