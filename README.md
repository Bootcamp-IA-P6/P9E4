# 🛡️ YouTube Hate Speech Detector - NPL Project

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.19-blue)](https://mlflow.org/)
[![Tests](https://img.shields.io/badge/Tests-31%20passing-green)]()

Solución de moderación automática de comentarios de YouTube basada en Machine Learning y Deep Learning. Detecta mensajes de odio en tiempo real mediante una interfaz web interactiva.

---

## 👥 Equipo

| Nombre | Rol |
|---|---|
| Camila | Data Scientist / ML Engineer |
| Mar | Data Scientist / ML Engineer |
| Michelle | Data Scientist / ML Engineer |
| JJ | Data Scientist / ML Engineer |

**Bootcamp IA — Factoria F5, Madrid**

---

## 🎯 Problema

YouTube recibe millones de comentarios diarios. Un equipo de moderadores humanos no puede escalar al ritmo de crecimiento de la plataforma. Este proyecto automatiza la detección de mensajes de odio para permitir una moderación eficiente a escala.

---

## 🚀 Demo

> 🔗 **App desplegada:** `[proximamente]`

```bash
# Ejecutar en local
cd app
uv run streamlit run app.py
```

---

## 📁 Estructura del proyecto

```
P9E4/
├── app/                        # Aplicación Streamlit
│   ├── app.py                  # Página principal
│   ├── pages/
│   │   ├── 1_texto.py          # Análisis de comentario individual
│   │   ├── 2_youtube.py        # Análisis por URL + monitoreo real-time
│   │   ├── 3_feedback.py       # Feedback de operadores
│   │   └── 4_dashboard.py      # Dashboard y estadísticas
│   └── utils/
│       ├── model.py            # Carga y predicción de modelos
│       ├── preprocessing.py    # Pipeline de limpieza de texto
│       ├── database.py         # Conexión a Supabase
│       ├── sidebar.py          # Sidebar compartido
│       └── ui.py               # Componentes UI reutilizables
├── data/
│   ├── raw/                    # Dataset original y enriquecido
│   └── vectorized/             # Matrices TF-IDF serializadas
├── models/
│   ├── model_v1/               # Ensemble Optuna + LSTM
│   └── model_v3/               # Logistic Regression (modelo final)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploración de datos
│   ├── 02_preprocessing.ipynb  # Preprocesamiento y augmentación
│   ├── 03_models.ipynb         # Modelos ML clásicos
│   ├── 04_lstm.ipynb           # Red neuronal LSTM
│   ├── 05_mlflow.ipynb         # Tracking de experimentos
│   └── V3_bert/
│       └── 01_bert_toxicity.ipynb  # Fine-tuning DistilBERT (Colab)
├── tests/
│   ├── test_pipeline.py        # Tests del pipeline NLP (31 tests)
│   └── test_streamlit.py       # Tests de la app
├── .env.example                # Variables de entorno ejemplo
├── docker-compose.yml          # Configuración Docker
├── Dockerfile                  # Imagen Docker
└── pyproject.toml              # Dependencias del proyecto
```

---

## 🤖 Modelos

Se entrenaron y compararon 8 configuraciones distintas, trackeadas con MLflow:

| Modelo | Dataset | F1 Test | Overfitting |
|---|---|---|---|
| Logistic Regression baseline | 1k | 0.750 | 0.188 ⚠️ |
| Voting Ensemble (LR+SVM+XGB) Optuna | 1k | 0.757 | **0.035 ✅** |
| LSTM Bidireccional | 1k | 0.628 | 0.293 ⚠️ |
| Logistic Regression balanced | 10k | 0.910 | 0.038 ✅ |
| LSTM Bidireccional (enriquecido) | 10k | 0.899 | 0.076 ⚠️ |
| **DistilBERT fine-tuned** | **10k** | **0.955** | **0.028 ✅** |

**Modelo seleccionado para producción:** Logistic Regression con `class_weight='balanced'` — mejor equilibrio entre rendimiento y eficiencia sin GPU.

**Modelo más preciso:** DistilBERT fine-tuned — F1=0.955, requiere GPU para inferencia eficiente.

---

## 🔬 Pipeline NLP

```
Texto original
    ↓
Limpieza con Regex (URLs, menciones, caracteres especiales)
    ↓
Lowercase
    ↓
Eliminación de stopwords (NLTK)
    ↓
Lematización (WordNetLemmatizer)
    ↓
Vectorización TF-IDF (500 features, bigrams, min_df=5)
    ↓
Clasificación (modelo seleccionado)
    ↓
Predicción + probabilidad
```

---

## 📊 Dataset

| Aspecto | Detalle |
|---|---|
| Dataset original | 1.000 comentarios de YouTube etiquetados |
| Dataset enriquecido | 10.414 comentarios (3 fuentes) |
| Balance de clases | 51% no tóxico / 49% tóxico |
| Idioma | Inglés (92%) |
| Fuentes | YouTube API, Kaggle Jigsaw, etiquetado manual |

**Augmentación aplicada:**
- Back translation (EN→ES→EN) sobre clase minoritaria
- Evaluada y descartada para el modelo final por incrementar overfitting

---

## 🖥️ Aplicación

La app tiene 4 páginas:

**🔍 Análisis de texto** — Introduce un comentario y obtén la predicción con probabilidad. Permite comparar todos los modelos simultáneamente.

**🎥 Análisis por URL** — Pega la URL de un vídeo de YouTube para analizar todos sus comentarios. Incluye monitoreo en tiempo real con alertas de contenido tóxico nuevo.

**💬 Feedback** — Los operadores pueden marcar predicciones incorrectas para mejorar el modelo en el futuro.

**📊 Dashboard** — Estadísticas agregadas, historial de predicciones y métricas del sistema.

---

## ⚙️ Instalación

### Requisitos
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/Bootcamp-IA-P6/P9E4.git
cd P9E4

# 2. Instalar dependencias
uv sync

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# 4. Ejecutar la app
cd app
uv run streamlit run app.py
```

### Variables de entorno

```env
SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJ...
YOUTUBE_API_KEY=AIzaSy...
```

---

## 🐳 Docker

```bash
# Construir imagen
docker build -t hate-speech-detector .

# Ejecutar
docker-compose up
```

---

## 🧪 Tests

```bash
# Ejecutar todos los tests
uv run pytest tests/ -v

# Solo tests del pipeline NLP
uv run pytest tests/test_pipeline.py -v

# Solo tests de la app
uv run pytest tests/test_streamlit.py -v
```

**31 tests** cubriendo:
- Pipeline de preprocesamiento (casos normales y edge cases)
- Carga y predicción de modelos (LogReg, Ensemble, LSTM)
- Integridad del dataset
- Vectorización TF-IDF
- Integración preprocessing + modelo

---

## 📈 MLflow

```bash
# Lanzar interfaz de MLflow
uv run mlflow ui --backend-store-uri ./mlruns

# Abrir en navegador
# http://127.0.0.1:5000
```

9 experimentos registrados con métricas completas (F1, precision, recall, overfitting gap).

---

## 🗄️ Base de datos

Supabase (PostgreSQL) almacena:
- **predictions** — cada predicción realizada (texto, modelo, probabilidad, fuente)
- **feedback** — correcciones de los operadores sobre predicciones incorrectas

---

## 🛠️ Tecnologías

| Categoría | Tecnología |
|---|---|
| Lenguaje | Python 3.11 |
| ML clásico | scikit-learn, XGBoost, LightGBM |
| Deep Learning | PyTorch, HuggingFace Transformers |
| NLP | NLTK, spaCy |
| Optimización | Optuna |
| Tracking | MLflow |
| App | Streamlit |
| Base de datos | Supabase (PostgreSQL) |
| YouTube API | google-api-python-client |
| Contenedores | Docker |
| Gestión deps | uv |
| Tests | pytest |

---

## 📋 Niveles de entrega cubiertos

- ✅ **Esencial** — Modelo ML, overfitting < 5%, interfaz Streamlit, Git organizado
- ✅ **Medio** — Ensemble, Optuna, tests unitarios, análisis por URL de YouTube
- ✅ **Avanzado** — LSTM, monitoreo real-time, Docker, despliegue público
- ✅ **Experto** — DistilBERT transformer, Supabase, MLflow tracking