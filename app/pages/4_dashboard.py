import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent))

from utils.database import get_predictions, get_feedback
from utils.sidebar import render_sidebar
from utils.ui import page_header, empty_state, color_by_verdict, COLOR_TOXIC, COLOR_NONTOXIC, COLOR_UNCERTAIN

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

render_sidebar()

page_header(
    "Dashboard",
    "Estadísticas agregadas de las predicciones y el modelo.",
    icon="📊",
)


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E6E9EF"),
    margin=dict(l=10, r=10, t=30, b=10),
)


@st.cache_data(ttl=60)
def load_data():
    preds = get_predictions(limit=1000)
    fb = get_feedback()
    return preds, fb


try:
    preds, fb = load_data()
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

if not preds:
    empty_state(
        "📊",
        "Todavía no hay datos",
        "Genera predicciones desde *Análisis de texto* o *Análisis por URL* "
        "y volverán a aparecer aquí.",
    )
    st.stop()

df = pd.DataFrame(preds)
df["created_at"] = pd.to_datetime(df["created_at"])
df["date"] = df["created_at"].dt.date

# ---------- KPIs ----------
total = len(df)
toxic = int((df["prediction"] == 1).sum())
nontoxic = total - toxic
toxic_pct = toxic / total * 100
today = df[df["date"] == pd.Timestamp.utcnow().date()]

st.subheader("Indicadores")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total predicciones", total)
c2.metric("🔴 Tóxicos", f"{toxic} ({toxic_pct:.1f}%)")
c3.metric("🟢 No tóxicos", nontoxic)
c4.metric("Hoy", len(today))
c5.metric("Feedback recibido", len(fb))

st.divider()

# ---------- Distribuciones ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución por clase")
    fig = go.Figure(
        data=[go.Pie(
            labels=["No tóxico", "Tóxico"],
            values=[nontoxic, toxic],
            hole=0.55,
            marker=dict(colors=[COLOR_NONTOXIC, COLOR_TOXIC]),
            textinfo="label+percent",
        )]
    )
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Predicciones por fuente")
    src = df["source"].value_counts().reset_index()
    src.columns = ["Fuente", "Cantidad"]
    fig = px.bar(
        src, x="Fuente", y="Cantidad",
        color="Fuente",
        color_discrete_sequence=[COLOR_TOXIC, COLOR_NONTOXIC, COLOR_UNCERTAIN],
        text="Cantidad",
    )
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=320)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Serie temporal ----------
st.subheader("Predicciones a lo largo del tiempo")
daily = (
    df.groupby(["date", "prediction"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "No tóxico", 1: "Tóxico"})
    .reset_index()
)
for col in ["No tóxico", "Tóxico"]:
    if col not in daily.columns:
        daily[col] = 0

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily["date"], y=daily["No tóxico"],
    mode="lines+markers", name="No tóxico",
    line=dict(color=COLOR_NONTOXIC, width=2),
    fill="tozeroy", fillcolor="rgba(78, 205, 196, 0.12)",
))
fig.add_trace(go.Scatter(
    x=daily["date"], y=daily["Tóxico"],
    mode="lines+markers", name="Tóxico",
    line=dict(color=COLOR_TOXIC, width=2),
    fill="tozeroy", fillcolor="rgba(255, 107, 107, 0.15)",
))
fig.update_layout(**PLOTLY_LAYOUT, height=320, legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

# ---------- Distribución de probabilidades ----------
st.subheader("Distribución de probabilidades")
fig = px.histogram(
    df, x="probability", nbins=20,
    color_discrete_sequence=[COLOR_TOXIC],
)
fig.update_layout(**PLOTLY_LAYOUT, height=300, bargap=0.05,
                  xaxis_title="Probabilidad de toxicidad", yaxis_title="Comentarios")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Feedback ----------
if fb:
    st.subheader("Calidad del modelo según feedback")
    df_fb = pd.DataFrame(fb)
    correct = int(df_fb["is_correct"].sum())
    incorrect = len(df_fb) - correct
    acc = correct / len(df_fb) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Total feedback", len(df_fb))
    c2.metric("✅ Predicciones correctas", correct)
    c3.metric("Tasa de acierto real", f"{acc:.1f}%")

    fig = go.Figure(
        data=[go.Pie(
            labels=["Correctas", "Incorrectas"],
            values=[correct, incorrect],
            hole=0.55,
            marker=dict(colors=[COLOR_NONTOXIC, COLOR_TOXIC]),
            textinfo="label+percent",
        )]
    )
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)
else:
    empty_state(
        "💬",
        "Sin feedback todavía",
        "Ve a la página *Feedback* para empezar a corregir predicciones.",
    )

st.divider()

# ---------- Tabla últimas predicciones ----------
st.subheader("Últimas predicciones")
df_show = df.head(20).copy()
df_show["Predicción"] = df_show["prediction"].map({0: "🟢 No tóxico", 1: "🔴 Tóxico"})
df_show["Probabilidad"] = (df_show["probability"] * 100).round(1).astype(str) + "%"
df_show["Fecha"] = df_show["created_at"].dt.strftime("%Y-%m-%d %H:%M")
df_show["Comentario"] = df_show["text"].str.slice(0, 80) + df_show["text"].apply(
    lambda t: "..." if len(t) > 80 else ""
)
df_show["_verdict"] = df_show["prediction"].map({0: "nontoxic", 1: "toxic"})

display_cols = ["Comentario", "Predicción", "Probabilidad", "source", "Fecha"]
df_show = df_show.rename(columns={"source": "Fuente"})
display_cols = ["Comentario", "Predicción", "Probabilidad", "Fuente", "Fecha"]

styled = df_show.style.apply(color_by_verdict("_verdict"), axis=1)
st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    column_order=display_cols,
)

st.divider()

# ---------- Métricas estáticas del modelo ----------
st.subheader("Métricas del modelo (entrenamiento)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("F1 Score", "0.909")
c2.metric("Overfitting gap", "0.039")
c3.metric("Modelo", "Logistic Regression")
c4.metric("Dataset", "10.414 comentarios")
