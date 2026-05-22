import streamlit as st

from utils.model import MODELS, DEFAULT_MODEL, model_options


def render_sidebar():
    """Sidebar común a todas las páginas: selector de modelo + info."""
    if "active_model" not in st.session_state:
        st.session_state["active_model"] = DEFAULT_MODEL

    with st.sidebar:
        st.markdown("### 🛡️ Hate Speech Detector")

        options = model_options()
        keys = list(options.keys())
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
