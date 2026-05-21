import pytest
import sys
import os
sys.path.append(os.path.abspath('..'))

import joblib
import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\r\n|\r|\n', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(t) for t in tokens
        if t not in stop_words
        and len(t) > 2
    ]
    return ' '.join(tokens)


# --- Tests de preprocesamiento ---
def test_preprocess_lowercase():
    assert preprocess("HELLO WORLD") == "hello world"

def test_preprocess_removes_urls():
    assert "http" not in preprocess("check http://example.com now")

def test_preprocess_removes_mentions():
    assert "@user" not in preprocess("hello @user how are you doing")

def test_preprocess_removes_special_chars():
    result = preprocess("hello!!! world??? 123")
    assert "!" not in result
    assert "?" not in result

def test_preprocess_removes_stopwords():
    result = preprocess("this is a very bad comment")
    assert "this" not in result
    assert "is" not in result

def test_preprocess_empty_string():
    assert preprocess("") == ""

def test_preprocess_none():
    assert preprocess(None) == ""

def test_preprocess_returns_string():
    assert isinstance(preprocess("hello world"), str)

# --- Tests del modelo ---
def test_model_loads():
    model = joblib.load('../models/model_v1/ensemble_final.pkl')
    assert model is not None

def test_tfidf_loads():
    tfidf = joblib.load('../models/model_v1/tfidf_final.pkl')
    assert tfidf is not None

def test_model_predicts_binary():
    model = joblib.load('../models/model_v1/ensemble_final.pkl')
    tfidf = joblib.load('../models/model_v1/tfidf_final.pkl')
    text = preprocess("you are a racist idiot")
    vector = tfidf.transform([text])
    pred = model.predict(vector)
    assert pred[0] in [0, 1]

def test_model_predict_proba():
    model = joblib.load('../models/model_v1/ensemble_final.pkl')
    tfidf = joblib.load('../models/model_v1/tfidf_final.pkl')
    text = preprocess("you are a racist idiot")
    vector = tfidf.transform([text])
    proba = model.predict_proba(vector)
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-6  # suman 1

def test_toxic_comment_detected():
    model = joblib.load('../models/model_v1/ensemble_final.pkl')
    tfidf = joblib.load('../models/model_v1/tfidf_final.pkl')
    text = preprocess("i hate you racist black people kill them")
    vector = tfidf.transform([text])
    pred = model.predict(vector)
    assert pred[0] == 1  # debe ser tóxico

def test_clean_comment_detected():
    model = joblib.load('../models/model_v1/ensemble_final.pkl')
    tfidf = joblib.load('../models/model_v1/tfidf_final.pkl')
    text = preprocess("this is a beautiful sunny day i love nature")
    vector = tfidf.transform([text])
    pred = model.predict(vector)
    assert pred[0] == 0  # debe ser no tóxico

# --- Tests del dataset ---
def test_dataset_loads():
    df = pd.read_csv('../data/processed/comments_processed.csv')
    assert len(df) > 0

def test_dataset_columns():
    df = pd.read_csv('../data/processed/comments_processed.csv')
    assert 'Text' in df.columns
    assert 'IsToxic' in df.columns
    assert 'text_clean' in df.columns

def test_dataset_no_nulls():
    df = pd.read_csv('../data/processed/comments_processed.csv')
    assert df['text_clean'].isna().sum() == 0

def test_dataset_binary_labels():
    df = pd.read_csv('../data/processed/comments_processed.csv')
    assert set(df['IsToxic'].unique()).issubset({0, 1})
    
# --- Tests de preprocesamiento edge cases ---
def test_preprocess_only_numbers():
    assert preprocess("123 456 789") == ""

def test_preprocess_only_special_chars():
    assert preprocess("!!! ??? ###") == ""

def test_preprocess_very_long_text():
    text = "hate " * 1000
    result = preprocess(text)
    assert isinstance(result, str)

def test_preprocess_line_breaks():
    result = preprocess("hello\r\nworld")
    assert "\r" not in result
    assert "\n" not in result

def test_preprocess_multiple_spaces():
    result = preprocess("hello     world")
    assert "  " not in result

# --- Tests del modelo con casos límite ---
def test_model_empty_text():
    model = joblib.load('../models/model_v1/ensemble_final.pkl')
    tfidf = joblib.load('../models/model_v1/tfidf_final.pkl')
    text = preprocess("")
    vector = tfidf.transform([text])
    pred = model.predict(vector)
    assert pred[0] in [0, 1]

def test_model_multiple_comments():
    model = joblib.load('../models/model_v1/ensemble_final.pkl')
    tfidf = joblib.load('../models/model_v1/tfidf_final.pkl')
    texts = [
        preprocess("i hate you"),
        preprocess("beautiful day today"),
        preprocess("racist idiot go away")
    ]
    vectors = tfidf.transform(texts)
    preds = model.predict(vectors)
    assert len(preds) == 3
    assert all(p in [0, 1] for p in preds)

def test_model_proba_between_0_and_1():
    model = joblib.load('../models/model_v1/ensemble_final.pkl')
    tfidf = joblib.load('../models/model_v1/tfidf_final.pkl')
    text = preprocess("some random comment")
    vector = tfidf.transform([text])
    proba = model.predict_proba(vector)
    assert all(0 <= p <= 1 for p in proba[0])

# --- Tests del dataset enriquecido ---
def test_enriched_dataset_loads():
    df = pd.read_csv('../data/raw/dataset_enriquecido.csv')
    assert len(df) > 1000

def test_enriched_dataset_balance():
    df = pd.read_csv('../data/raw/dataset_enriquecido.csv')
    balance = df['IsToxic'].value_counts(normalize=True)
    assert balance.max() < 0.60

def test_enriched_dataset_no_duplicates():
    df = pd.read_csv('../data/raw/dataset_enriquecido.csv')
    assert df.duplicated().sum() == 0

# --- Tests de vectorización ---
def test_tfidf_output_shape():
    tfidf = joblib.load('../models/model_v1/tfidf_final.pkl')
    texts = ["hello world", "racist comment here"]
    vectors = tfidf.transform(texts)
    assert vectors.shape[0] == 2
    assert vectors.shape[1] == tfidf.vocabulary_.__len__()

def test_tfidf_unknown_words():
    tfidf = joblib.load('../models/model_v1/tfidf_final.pkl')
    text = ["xyzabcdefgh unknown gibberish words"]
    vector = tfidf.transform(text)
    assert vector.shape == (1, tfidf.vocabulary_.__len__())