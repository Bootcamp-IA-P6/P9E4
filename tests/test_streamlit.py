import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))

from utils.preprocessing import preprocess
from utils.model import load_model, predict, label_for, MODELS, DEFAULT_MODEL

# --- Tests de carga de modelos ---
def test_default_model_exists():
    assert DEFAULT_MODEL in MODELS

def test_all_models_have_required_keys():
    required_keys = ['display', 'description', 'type']
    for name, info in MODELS.items():
        for key in required_keys:
            assert key in info, f"Modelo {name} no tiene clave {key}"

def test_logreg_loads():
    model, tfidf = load_model('logreg_v3')
    assert model is not None
    assert tfidf is not None

def test_ensemble_loads():
    model, tfidf = load_model('ensemble_v1')
    assert model is not None
    assert tfidf is not None

def test_lstm_loads():
    model, vocab = load_model('lstm_enriched')
    assert model is not None
    assert vocab is not None

# --- Tests de predicción ---
def test_logreg_predict_returns_tuple():
    model, tfidf = load_model('logreg_v3')
    pred, proba = predict(preprocess("you are a racist idiot"), model, tfidf)
    assert isinstance(pred, int)
    assert isinstance(proba, float)
    assert pred in [0, 1]
    assert 0 <= proba <= 1

def test_lstm_predict_returns_tuple():
    model, vocab = load_model('lstm_enriched')
    pred, proba = predict(preprocess("you are a racist idiot"), model, vocab)
    assert isinstance(pred, int)
    assert isinstance(proba, float)
    assert pred in [0, 1]
    assert 0 <= proba <= 1

def test_predict_empty_text():
    model, tfidf = load_model('logreg_v3')
    pred, proba = predict('', model, tfidf)
    assert pred in [0, 1]
    assert 0 <= proba <= 1

def test_predict_proba_sums_to_one():
    model, tfidf = load_model('logreg_v3')
    text = preprocess("this is a test comment")
    vector = tfidf.transform([text])
    proba = model.predict_proba(vector)[0]
    assert abs(sum(proba) - 1.0) < 1e-6

# --- Tests de label_for ---
def test_label_for_toxic():
    assert label_for(0.9) == "toxic"

def test_label_for_nontoxic():
    assert label_for(0.1) == "nontoxic"

def test_label_for_uncertain():
    assert label_for(0.5) == "uncertain"

def test_label_for_boundary_high():
    assert label_for(0.60) == "toxic"

def test_label_for_boundary_low():
    assert label_for(0.40) == "nontoxic"

# --- Tests de model_options ---
def test_model_options_returns_dict():
    from utils.model import model_options
    options = model_options()
    assert isinstance(options, dict)
    assert len(options) > 0

def test_model_options_has_all_models():
    from utils.model import model_options
    options = model_options()
    for key in MODELS:
        assert key in options

# --- Tests de integración preprocessing + modelo ---
def test_pipeline_toxic_comment():
    model, tfidf = load_model('logreg_v3')
    text = preprocess("you are a disgusting racist idiot kill yourself")
    pred, proba = predict(text, model, tfidf)
    assert pred == 1

def test_pipeline_clean_comment():
    model, tfidf = load_model('logreg_v3')
    text = preprocess("this is a beautiful sunny day i love nature")
    pred, proba = predict(text, model, tfidf)
    assert pred == 0