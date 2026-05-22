import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import preprocess
from utils.model import load_model, predict, label_for, MODELS
from utils.database import save_prediction
from utils.sidebar import render_sidebar
from utils.ui import page_header

st.set_page_config(page_title="Análisis de texto", page_icon="🔍", layout="wide")

active = render_sidebar()

page_header(
    "Análisis de comentario",
    "Introduce un comentario para detectar si contiene lenguaje de odio.",
    icon="🔍",
)


@st.cache_resource
def get_model(name: str):
    return load_model(name)

# Input
text_input = st.text_area(
    "Comentario a analizar",
    placeholder="Escribe o pega aquí el comentario de YouTube...",
    height=150,
)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    analyze = st.button("🔍 Analizar", type="primary", use_container_width=True)
with col2:
    clear = st.button("🗑️ Limpiar", use_container_width=True)
with col3:
    compare = st.checkbox(
        "Comparar todos los modelos",
        value=False,
        help="Ejecuta cada modelo y muestra los resultados lado a lado",
    )

if clear:
    st.rerun()

if analyze and text_input.strip():
    text_clean = preprocess(text_input)

    if compare:
        # ----- Modo comparación -----
        st.divider()
        st.subheader("Comparación entre modelos")
        rows = []
        with st.spinner("Ejecutando todos los modelos..."):
            for name, info in MODELS.items():
                model, tfidf = get_model(name)
                pred, proba = predict(text_clean, model, tfidf)
                verdict = label_for(proba)
                pretty = {
                    "toxic":     "🔴 Tóxico",
                    "nontoxic":  "🟢 No tóxico",
                    "uncertain": "🟡 Incierto",
                }[verdict]
                rows.append({
                    "Modelo":              info["display"],
                    "Predicción":          pretty,
                    "Probabilidad tóxico": f"{proba * 100:.1f}%",
                    "_pred":               pred,
                    "_proba":              proba,
                    "_key":                name,
                })

        st.dataframe(
            [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
            use_container_width=True,
            hide_index=True,
        )

        preds = {r["_pred"] for r in rows}
        if len(preds) == 1:
            st.success("✅ Los modelos coinciden en la predicción.")
        else:
            st.warning("⚠️ Los modelos discrepan — revisa los detalles.")

        # Texto procesado
        with st.expander("Ver texto preprocesado"):
            st.code(text_clean)

        # En modo compare guardamos la predicción del modelo activo
        active_row = next(r for r in rows if r["_key"] == active)
        try:
            pred_id = save_prediction(
                text=text_input,
                text_clean=text_clean,
                prediction=active_row["_pred"],
                probability=active_row["_proba"],
                source="manual",
            )
            st.session_state["last_prediction_id"] = pred_id
            st.caption(
                f"✅ Predicción del modelo activo ({MODELS[active]['display']}) "
                f"guardada — ID: `{pred_id}`"
            )
        except Exception as e:
            st.caption(f"⚠️ No se pudo guardar: {e}")

    else:
        # ----- Modo single -----
        with st.spinner(f"Analizando con {MODELS[active]['display']}..."):
            model, tfidf = get_model(active)
            pred, proba = predict(text_clean, model, tfidf)

        st.divider()
        st.caption(f"Modelo usado: **{MODELS[active]['display']}**")

        verdict = label_for(proba)
        confidence_pct = (proba if pred == 1 else 1 - proba) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            if verdict == "toxic":
                st.error("🔴 TÓXICO")
            elif verdict == "nontoxic":
                st.success("🟢 NO TÓXICO")
            else:
                st.warning("🟡 INCIERTO")
        with col2:
            st.metric("Probabilidad tóxico", f"{proba * 100:.1f}%")
        with col3:
            st.metric("Confianza en la clase", f"{confidence_pct:.1f}%")

        if verdict == "uncertain":
            st.info(
                "El modelo no está seguro (probabilidad entre 40% y 60%). "
                "Trátalo como un caso ambiguo."
            )

        with st.expander("Ver texto preprocesado"):
            st.code(text_clean)

        try:
            pred_id = save_prediction(
                text=text_input,
                text_clean=text_clean,
                prediction=pred,
                probability=proba,
                source="manual",
            )
            st.session_state["last_prediction_id"] = pred_id
            st.caption(f"✅ Predicción guardada — ID: `{pred_id}`")
        except Exception as e:
            st.caption(f"⚠️ No se pudo guardar: {e}")

elif analyze and not text_input.strip():
    st.warning("Por favor introduce un comentario.")
