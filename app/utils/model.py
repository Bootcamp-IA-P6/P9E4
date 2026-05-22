import joblib
<<<<<<< HEAD
import torch
import torch.nn as nn
import numpy as np
=======
>>>>>>> 4d1cc9fa50a03ef6012b7c56a662e009e32e91ae
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).parent.parent.parent

<<<<<<< HEAD
# --- Arquitectura LSTM (debe ser idéntica a la del notebook) ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden).squeeze(1)


MODELS = {
    "logreg_v3": {
        "display":     "Logistic Regression (v3)",
        "model_path":  ROOT / "models" / "model_v3" / "logreg_model.pkl",
        "tfidf_path":  ROOT / "models" / "model_v3" / "tfidf_vectorizer.pkl",
        "description": "Baseline rápido. F1 0.909, gap 0.039.",
        "type":        "sklearn"
    },
    "ensemble_v1": {
        "display":     "Voting Ensemble (v1)",
        "model_path":  ROOT / "models" / "model_v1" / "ensemble_final.pkl",
        "tfidf_path":  ROOT / "models" / "model_v1" / "tfidf_final.pkl",
        "description": "Combinación de varios clasificadores. Más robusto pero más lento.",
        "type":        "sklearn"
    },
    "lstm_enriched": {
        "display":     "LSTM Bidireccional (dataset enriquecido)",
        "model_path":  ROOT / "models" / "model_v1" / "lstm_enriched.pt",
        "vocab_path":  ROOT / "models" / "model_v1" / "lstm_vocab_enriched.pt",
        "description": "Red neuronal LSTM. F1 0.899, dataset 10k comentarios.",
        "type":        "lstm"
    },
    "distilbert": {
        "display":     "DistilBERT (transformer)",
        "hf_model":    "Michh14/youtube-hate-speech-distilbert",
        "description": "Transformer fine-tuned. F1 0.955, overfitting 0.028. Más lento.",
        "type":        "bert",
        "threshold": 0.35
    },
}

DEFAULT_MODEL = "logreg_v3"

UNCERTAIN_LOW  = 0.40
UNCERTAIN_HIGH = 0.60

MAX_LEN = 100
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(name: str = DEFAULT_MODEL):
    if name not in MODELS:
        raise ValueError(f"Modelo desconocido: {name}. Opciones: {list(MODELS)}")
    info = MODELS[name]

    if info["type"] == "bert":
        tokenizer = AutoTokenizer.from_pretrained(info["hf_model"])
        model = AutoModelForSequenceClassification.from_pretrained(info["hf_model"])
        model.eval()
        model.to(device)
        return model, tokenizer

    elif info["type"] == "lstm":
        vocab = torch.load(info["vocab_path"], map_location=device, weights_only=False)
        model = LSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=100, hidden_dim=128,
            n_layers=2, dropout=0.3
        ).to(device)
        model.load_state_dict(torch.load(info["model_path"], map_location=device, weights_only=False))
        model.eval()
        return model, vocab

    else:
        model = joblib.load(info["model_path"])
        tfidf = joblib.load(info["tfidf_path"])
        return model, tfidf


def _text_to_indices(text: str, vocab: dict, max_len: int = MAX_LEN) -> list:
    tokens  = text.split()[:max_len]
    indices = [vocab.get(t, 1) for t in tokens]
    indices += [0] * (max_len - len(indices))
    return indices


def predict(text_clean: str, model, tfidf_or_vocab, model_type: str = None) -> tuple[int, float]:
    
    # Detectar tipo por clase
    if hasattr(tfidf_or_vocab, 'encode_plus') or 'Tokenizer' in type(tfidf_or_vocab).__name__:
        # BERT
        tokenizer = tfidf_or_vocab
        encoding = tokenizer(
            text_clean,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            output = model(**encoding)
        proba = torch.softmax(output.logits, dim=1)[0]
        proba_toxic = float(proba[1].item())
        threshold = MODELS.get('distilbert', {}).get('threshold', 0.5)
        pred = int(proba_toxic >= threshold)
        return pred, proba_toxic

    elif isinstance(model, LSTMClassifier):
        vocab = tfidf_or_vocab
        indices = _text_to_indices(text_clean, vocab)
        tensor = torch.tensor([indices], dtype=torch.long).to(device)
        with torch.no_grad():
            logit = model(tensor)
            proba_toxic = float(torch.sigmoid(logit).item())
        pred = int(proba_toxic >= 0.5)
        return pred, proba_toxic

    else:
        vector = tfidf_or_vocab.transform([text_clean])
        proba = model.predict_proba(vector)[0]
        classes = list(model.classes_)
        toxic_idx = classes.index(max(classes))
        proba_toxic = float(proba[toxic_idx])
        pred = int(proba_toxic >= 0.5)
        return pred, proba_toxic


def label_for(proba_toxic: float) -> str:
=======
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
>>>>>>> 4d1cc9fa50a03ef6012b7c56a662e009e32e91ae
    if proba_toxic >= UNCERTAIN_HIGH:
        return "toxic"
    if proba_toxic <= UNCERTAIN_LOW:
        return "nontoxic"
    return "uncertain"


def model_options() -> dict[str, str]:
<<<<<<< HEAD
    return {key: info["display"] for key, info in MODELS.items()}
=======
    return {key: info["display"] for key, info in MODELS.items()}
>>>>>>> 4d1cc9fa50a03ef6012b7c56a662e009e32e91ae
