import streamlit as st
from pathlib import Path
from utils.model import MODELS, DEFAULT_MODEL, model_options

def render_sidebar():
    """Sidebar común a todas las páginas: selector de modelo + info."""
    if "active_model" not in st.session_state:
        st.session_state["active_model"] = DEFAULT_MODEL

    with st.sidebar:
        st.markdown("### 🛡️ Hate Speech Detector")

        # Navegación
        st.markdown("#### Navegación")
        if st.button("🏠 Inicio",              use_container_width=True):
            st.switch_page("app.py")
        if st.button("🔍 Análisis de texto",   use_container_width=True):
            st.switch_page("pages/1_texto.py")
        if st.button("🎥 Análisis por URL",    use_container_width=True):
            st.switch_page("pages/2_youtube.py")
        if st.button("💬 Feedback",            use_container_width=True):
            st.switch_page("pages/3_feedback.py")
        if st.button("📊 Dashboard",           use_container_width=True):
            st.switch_page("pages/4_dashboard.py")

        st.divider()

        # Selector de modelo
        options = model_options()
        keys    = list(options.keys())
        current = st.session_state["active_model"]
        selected = st.selectbox(
            "Modelo activo",
            keys,
            index=keys.index(current) if current in keys else 0,
            format_func=lambda k: options[k],
            key="model_selector",
        )
        st.session_state["active_model"] = selected
        st.caption(MODELS[selected]["description"])

        st.divider()
        st.markdown("**Dataset:** 10.414 comentarios")
        st.markdown("**F1 Score:** 0.909")
        st.markdown("**Overfitting gap:** 0.039")
        st.divider()
        st.markdown("Factoria F5 — Madrid AI Bootcamp")

    return st.session_state["active_model"]
