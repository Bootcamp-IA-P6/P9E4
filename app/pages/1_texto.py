import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import preprocess
from utils.model import load_model, predict
from utils.database import save_prediction

st.set_page_config(page_title="Análisis de texto", page_icon="🔍", layout="wide")

st.title("🔍 Análisis de comentario")
st.markdown("Introduce un comentario para analizar si contiene lenguaje de odio.")

# Cargar modelo
@st.cache_resource
def get_model():
    return load_model()

model, tfidf = get_model()

# Input
text_input = st.text_area(
    "Comentario a analizar",
    placeholder="Escribe o pega aquí el comentario de YouTube...",
    height=150
)

col1, col2 = st.columns([1, 1])
with col1:
    analyze = st.button("🔍 Analizar", type="primary", use_container_width=True)
with col2:
    clear = st.button("🗑️ Limpiar", use_container_width=True)

if clear:
    st.rerun()

if analyze and text_input.strip():
    with st.spinner("Analizando..."):
        text_clean = preprocess(text_input)
        pred, proba = predict(text_clean, model, tfidf)

    st.divider()

    # Resultado
    col1, col2, col3 = st.columns(3)

    with col1:
        if pred == 1:
            st.error("🔴 TÓXICO")
        else:
            st.success("🟢 NO TÓXICO")

    with col2:
        st.metric("Probabilidad", f"{proba*100:.1f}%")

    with col3:
        st.metric("Confianza", "Alta" if proba > 0.8 else "Media" if proba > 0.6 else "Baja")

    # Texto procesado
    with st.expander("Ver texto preprocesado"):
        st.code(text_clean)

    # Guardar en Supabase
    try:
        pred_id = save_prediction(
            text=text_input,
            text_clean=text_clean,
            prediction=pred,
            probability=proba,
            source='manual'
        )
        st.session_state['last_prediction_id'] = pred_id
        st.caption(f"✅ Predicción guardada — ID: `{pred_id}`")
    except Exception as e:
        st.caption(f"⚠️ No se pudo guardar: {e}")

elif analyze and not text_input.strip():
    st.warning("Por favor introduce un comentario.")