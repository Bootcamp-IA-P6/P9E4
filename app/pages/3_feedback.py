import streamlit as st
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from utils.database import get_predictions, get_feedback, save_feedback, get_client
from utils.sidebar import render_sidebar
from utils.ui import page_header, empty_state

st.set_page_config(page_title="Feedback", page_icon="💬", layout="wide")

render_sidebar()

page_header(
    "Feedback de predicciones",
    "Corrige predicciones del modelo. Estas correcciones se usarán para reentrenar y mejorar la precisión.",
    icon="💬",
)


@st.cache_data(ttl=30)
def load_data(limit: int):
    preds = get_predictions(limit=limit)
    fb = get_feedback()
    return preds, fb


def submit_feedback(prediction_id: str, is_correct: bool, comment: str = None):
    save_feedback(prediction_id=prediction_id, is_correct=is_correct, comment=comment or None)
    st.cache_data.clear()


# ---------- Feedback rápido sobre la última predicción ----------
last_id = st.session_state.get("last_prediction_id")
if last_id:
    st.subheader("⚡ Feedback rápido")
    st.caption(f"Última predicción de esta sesión — ID: `{last_id}`")

    try:
        client = get_client()
        row = client.table("predictions").select("*").eq("id", last_id).single().execute().data
    except Exception:
        row = None

    if row:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"**Comentario:** {row['text']}")
            label = "🔴 Tóxico" if row["prediction"] == 1 else "🟢 No tóxico"
            st.markdown(f"**Predicción:** {label} ({row['probability'] * 100:.1f}%)")
        with col2:
            comment_quick = st.text_input("Comentario (opcional)", key="quick_comment")
            c1, c2 = st.columns(2)
            if c1.button("✅ Correcta", key="quick_ok", use_container_width=True):
                submit_feedback(last_id, True, comment_quick)
                st.success("Feedback guardado")
                st.session_state.pop("last_prediction_id", None)
                st.rerun()
            if c2.button("❌ Incorrecta", key="quick_nok", use_container_width=True):
                submit_feedback(last_id, False, comment_quick)
                st.success("Feedback guardado")
                st.session_state.pop("last_prediction_id", None)
                st.rerun()
    st.divider()


# ---------- Lista de predicciones ----------
st.subheader("📋 Predicciones recientes")

col1, col2 = st.columns([1, 2])
limit = col1.slider("Cuántas mostrar", 10, 100, 25, step=5)
only_pending = col2.checkbox("Solo predicciones sin feedback", value=True)

try:
    preds, fb = load_data(limit)
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

if not preds:
    empty_state(
        "📭",
        "Sin predicciones todavía",
        "Genera alguna desde *Análisis de texto* o *Análisis por URL* y volverás aquí para darles feedback.",
    )
    st.stop()

ids_with_fb = {f["prediction_id"] for f in fb}

df_preds = pd.DataFrame(preds)
if only_pending:
    df_preds = df_preds[~df_preds["id"].isin(ids_with_fb)]

if df_preds.empty:
    empty_state(
        "✨",
        "Todo al día",
        "No hay predicciones pendientes de feedback con los filtros actuales.",
    )
    st.stop()

st.caption(f"Mostrando {len(df_preds)} predicciones — {len(ids_with_fb)} ya con feedback.")

# Selector
options = {
    f"{r['text'][:80]}{'...' if len(r['text']) > 80 else ''}  →  "
    f"{'🔴' if r['prediction'] == 1 else '🟢'} {r['probability'] * 100:.0f}%": r["id"]
    for _, r in df_preds.iterrows()
}
selected_label = st.selectbox("Selecciona una predicción", list(options.keys()))
selected_id = options[selected_label]
row = df_preds[df_preds["id"] == selected_id].iloc[0]

# Detalle
st.divider()
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("**Texto original:**")
    st.code(row["text"], language=None)
    st.markdown("**Texto procesado:**")
    st.code(row["text_clean"], language=None)
with col2:
    label = "🔴 TÓXICO" if row["prediction"] == 1 else "🟢 NO TÓXICO"
    st.metric("Predicción", label)
    st.metric("Probabilidad", f"{row['probability'] * 100:.1f}%")
with col3:
    st.metric("Fuente", row["source"])
    if row.get("video_url"):
        st.caption(f"🎥 [Vídeo]({row['video_url']})")
    st.caption(f"📅 {row['created_at'][:19].replace('T', ' ')}")

st.divider()

# Form de feedback
already = selected_id in ids_with_fb
if already:
    st.warning("⚠️ Esta predicción ya tiene feedback. Si envías otro se añadirá como duplicado.")

with st.form("feedback_form", clear_on_submit=True):
    st.markdown("### ¿La predicción es correcta?")
    comment = st.text_area(
        "Comentario (opcional)",
        placeholder="Explica por qué crees que está mal, o cualquier nota útil...",
    )
    c1, c2 = st.columns(2)
    ok = c1.form_submit_button("✅ Es correcta", use_container_width=True)
    nok = c2.form_submit_button("❌ Es incorrecta", use_container_width=True)

if ok or nok:
    try:
        submit_feedback(selected_id, ok, comment)
        st.success("Feedback guardado. ¡Gracias!")
        st.rerun()
    except Exception as e:
        st.error(f"No se pudo guardar el feedback: {e}")

# Resumen
st.divider()
total_fb = len(fb)
if total_fb:
    correct = sum(1 for f in fb if f["is_correct"])
    incorrect = total_fb - correct
    acc = correct / total_fb * 100
    col1, col2, col3 = st.columns(3)
    col1.metric("Total feedback", total_fb)
    col2.metric("✅ Correctas", correct)
    col3.metric("Tasa de acierto", f"{acc:.1f}%")
