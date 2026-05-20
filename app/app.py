import streamlit as st

st.set_page_config(
    page_title="YouTube Hate Speech Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ YouTube Hate Speech Detector")
st.markdown("""
Herramienta de moderación automática de comentarios de YouTube 
basada en Machine Learning.

---

### ¿Cómo usar la app?

| Página | Descripción |
|---|---|
| 🔍 Análisis de texto | Analiza un comentario individual |
| 🎥 Análisis por URL | Analiza todos los comentarios de un vídeo |
| 💬 Feedback | Corrige predicciones incorrectas |
| 📊 Dashboard | Estadísticas y métricas del modelo |

---
""")

st.info("Selecciona una página en el menú de la izquierda para empezar.")

# Sidebar info
with st.sidebar:
    st.markdown("### 🛡️ Hate Speech Detector")
    st.markdown("**Modelo:** Logistic Regression")
    st.markdown("**Dataset:** 10.414 comentarios")
    st.markdown("**F1 Score:** 0.909")
    st.markdown("**Overfitting gap:** 0.039")
    st.divider()
    st.markdown("Factoria F5 — Madrid AI Bootcamp")