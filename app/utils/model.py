import joblib
import numpy as np
from pathlib import Path

MODEL_PATH  = Path(__file__).parent.parent.parent / 'models' / 'model_v3' / 'logreg_model.pkl'
TFIDF_PATH  = Path(__file__).parent.parent.parent / 'models' / 'model_v3' / 'tfidf_vectorizer.pkl'

def load_model():
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    return model, tfidf

def predict(text_clean: str, model, tfidf) -> tuple[int, float]:
    vector = tfidf.transform([text_clean])
    pred   = model.predict(vector)[0]
    proba  = model.predict_proba(vector)[0]
    return int(pred), float(max(proba))