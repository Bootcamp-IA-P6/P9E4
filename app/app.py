import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.sidebar import render_sidebar
from utils.ui import page_header

st.set_page_config(
    page_title="YouTube Hate Speech Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

page_header(
    "YouTube Hate Speech Detector",
    "Moderación automática de comentarios basada en Machine Learning.",
    icon="🛡️",
)

st.markdown(
    """
<style>
.card {
    background: linear-gradient(180deg, #1C2128 0%, #181C24 100%);
    border: 1px solid #2a2f3a;
    border-radius: 12px;
    padding: 1.2rem 1.3rem;
    height: 100%;
}
.card h4 { margin: .2rem 0 .5rem 0; color: #E6E9EF; }
.card p  { margin: 0; color: #9aa4b2; font-size: .92rem; line-height: 1.4; }
.card .icon { font-size: 1.7rem; }
</style>
""",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

cards = [
    ("🔍", "Análisis de texto", "Analiza un comentario individual con el modelo activo.", c1),
    ("🎥", "Análisis por URL", "Procesa todos los comentarios de un vídeo de YouTube.", c2),
    ("💬", "Feedback", "Corrige predicciones para mejorar el modelo en el futuro.", c3),
    ("📊", "Dashboard", "Estadísticas agregadas y métricas del modelo.", c4),
]

for icon, title, desc, col in cards:
    with col:
        st.markdown(
            f"""
            <div class="card">
                <div class="icon">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")
st.info("👈 Selecciona una página en el menú lateral para empezar.")
