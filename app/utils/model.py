import joblib
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

MODELS = {
    "logreg_v3": {
        "display":     "Logistic Regression (v3)",
        "model_path":  ROOT / "models" / "model_v3" / "logreg_model.pkl",
        "tfidf_path":  ROOT / "models" / "model_v3" / "tfidf_vectorizer.pkl",
        "description": "Baseline rápido. F1 0.909, gap 0.039.",
    },
    "ensemble_v1": {
        "display":     "Voting Ensemble (v1)",
        "model_path":  ROOT / "models" / "model_v1" / "ensemble_final.pkl",
        "tfidf_path":  ROOT / "models" / "model_v1" / "tfidf_final.pkl",
        "description": "Combinación de varios clasificadores. Más robusto pero más lento.",
    },
}

DEFAULT_MODEL = "logreg_v3"


def load_model(name: str = DEFAULT_MODEL):
    if name not in MODELS:
        raise ValueError(f"Modelo desconocido: {name}. Opciones: {list(MODELS)}")
    info = MODELS[name]
    model = joblib.load(info["model_path"])
    tfidf = joblib.load(info["tfidf_path"])
    return model, tfidf


UNCERTAIN_LOW = 0.40
UNCERTAIN_HIGH = 0.60


def predict(text_clean: str, model, tfidf) -> tuple[int, float]:
    """Devuelve (pred, proba_toxic). `proba_toxic` es siempre la probabilidad
    de la clase positiva (tóxico)."""
    vector = tfidf.transform([text_clean])
    proba = model.predict_proba(vector)[0]
    classes = list(model.classes_)
    toxic_idx = classes.index(max(classes))  # 1 o True
    proba_toxic = float(proba[toxic_idx])
    pred = int(proba_toxic >= 0.5)
    return pred, proba_toxic


def label_for(proba_toxic: float) -> str:
    """Devuelve 'toxic', 'nontoxic' o 'uncertain' según el margen."""
    if proba_toxic >= UNCERTAIN_HIGH:
        return "toxic"
    if proba_toxic <= UNCERTAIN_LOW:
        return "nontoxic"
    return "uncertain"


def model_options() -> dict[str, str]:
    return {key: info["display"] for key, info in MODELS.items()}
