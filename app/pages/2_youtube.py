import streamlit as st
import sys
from pathlib import Path
import os
import re
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import pandas as pd

load_dotenv(Path(__file__).parent.parent.parent / '.env')

sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import preprocess
from utils.model import load_model, predict, label_for, MODELS
from utils.database import save_prediction
from utils.sidebar import render_sidebar
from utils.ui import page_header, color_by_verdict, render_video_preview

st.set_page_config(page_title="Análisis por URL", page_icon="🎥", layout="wide")

active = render_sidebar()

page_header(
    "Análisis de comentarios por URL",
    "Introduce la URL de un vídeo de YouTube para analizar sus comentarios.",
    icon="🎥",
)

st.caption(f"Modelo activo: **{MODELS[active]['display']}** — cámbialo desde el sidebar.")


@st.cache_resource
def get_model(name: str):
    return load_model(name)


@st.cache_resource
def get_youtube():
    api_key = os.getenv('YOUTUBE_API_KEY')
    return build('youtube', 'v3', developerKey=api_key)


def extract_video_id(url: str) -> str | None:
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_comments(youtube, video_id: str, max_comments: int = 100) -> list:
    comments = []
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=min(max_comments, 100),
        textFormat='plainText',
    )
    while request and len(comments) < max_comments:
        response = request.execute()
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'text':   comment['textDisplay'],
                'author': comment['authorDisplayName'],
                'likes':  comment['likeCount'],
                'date':   comment['publishedAt'][:10],
            })
        request = youtube.commentThreads().list_next(request, response)
    return comments[:max_comments]


def pretty(verdict: str) -> str:
    return {"toxic": "🔴 Tóxico", "nontoxic": "🟢 No tóxico", "uncertain": "🟡 Incierto"}[verdict]


# --- UI ---
url_input = st.text_input(
    "URL del vídeo",
    placeholder="https://www.youtube.com/watch?v=...",
)

# Preview del vídeo cuando la URL es válida
video_info = None
if url_input.strip():
    vid = extract_video_id(url_input)
    if vid:
        try:
            video_info = render_video_preview(get_youtube(), vid)
        except Exception:
            video_info = None

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    max_comments = st.slider("Número máximo de comentarios", 10, 200, 50)
with col2:
    save_all = st.checkbox("Guardar predicciones en base de datos", value=True)
with col3:
    compare = st.checkbox(
        "Comparar todos los modelos",
        value=False,
        help="Ejecuta cada modelo sobre todos los comentarios. Más lento.",
    )

# Decidir si el botón se puede activar
can_analyze = True
block_msg = None
if video_info is not None:
    if video_info["comments_disabled"]:
        can_analyze = False
        block_msg = "🚫 Este vídeo tiene los comentarios desactivados por su creador. Prueba con otro vídeo."
    elif video_info["comment_count"] == 0:
        can_analyze = False
        block_msg = "💬 Este vídeo no tiene comentarios todavía. Prueba con otro."

if block_msg:
    st.warning(block_msg)

analyze = st.button(
    "🔍 Analizar comentarios",
    type="primary",
    disabled=not can_analyze,
)


# --- Lanzar análisis ---
if analyze:
    if not url_input.strip():
        st.warning("Por favor introduce una URL.")
    else:
        video_id = extract_video_id(url_input)
        if not video_id:
            st.error("URL no válida. Asegúrate de que es un enlace de YouTube.")
        else:
            try:
                youtube = get_youtube()
                model_keys = list(MODELS.keys()) if compare else [active]
                loaded = {name: get_model(name) for name in model_keys}

                with st.spinner("Obteniendo comentarios..."):
                    comments = get_comments(youtube, video_id, max_comments)

                if not comments:
                    st.warning("No se encontraron comentarios en este vídeo.")
                else:
                    info = st.empty()
                    info.info(
                        f"Analizando {len(comments)} comentarios con "
                        f"{len(loaded)} modelo{'s' if len(loaded) > 1 else ''}..."
                    )
                    progress = st.progress(0)
                    results = []

                    for i, comment in enumerate(comments):
                        text_clean = preprocess(comment['text'])
                        row = {
                            'Comentario': comment['text'],
                            'Autor':      comment['author'],
                            'Likes':      comment['likes'],
                            'Fecha':      comment['date'],
                            '_clean':     text_clean,
                        }
                        for name, (mdl, tfidf) in loaded.items():
                            pred, proba = predict(text_clean, mdl, tfidf)
                            verdict = label_for(proba)
                            display = MODELS[name]['display']
                            row[f'Resultado ({display})'] = pretty(verdict)
                            row[f'Prob. tóxico ({display})'] = f"{proba * 100:.1f}%"
                            row[f'_pred_{name}'] = pred
                            row[f'_proba_{name}'] = proba
                            row[f'_verdict_{name}'] = verdict

                        if save_all:
                            try:
                                save_prediction(
                                    text=comment['text'],
                                    text_clean=text_clean,
                                    prediction=row[f'_pred_{active}'],
                                    probability=row[f'_proba_{active}'],
                                    source='youtube',
                                    video_url=url_input,
                                )
                            except Exception:
                                pass

                        results.append(row)
                        progress.progress((i + 1) / len(comments))

                    progress.empty()
                    info.empty()

                    st.session_state['yt_results'] = pd.DataFrame(results)
                    st.session_state['yt_url'] = url_input
                    st.session_state['yt_models'] = model_keys
                    st.session_state['yt_compare'] = compare

            except HttpError as e:
                reason = ""
                try:
                    reason = e.error_details[0].get("reason", "") if e.error_details else ""
                except Exception:
                    pass
                msg = str(e)
                if reason == "commentsDisabled" or "commentsDisabled" in msg:
                    st.error("🚫 Este vídeo tiene los comentarios desactivados. Prueba con otro.")
                elif reason in ("quotaExceeded", "rateLimitExceeded"):
                    st.error("⏳ Se ha excedido la cuota de la YouTube API. Espera un rato e inténtalo de nuevo.")
                elif reason == "videoNotFound" or "videoNotFound" in msg:
                    st.error("❓ No se encontró el vídeo. ¿La URL es correcta?")
                else:
                    st.error(f"Error de YouTube: {reason or e.resp.status}")
            except Exception as e:
                st.error(f"Error al obtener comentarios: {e}")


# --- Mostrar resultados ---
df = st.session_state.get('yt_results')
if df is not None and not df.empty:
    st.divider()

    model_keys = st.session_state.get('yt_models', [active])
    is_compare = st.session_state.get('yt_compare', False)
    models_label = ", ".join(MODELS[k]['display'] for k in model_keys)
    st.caption(
        f"Resultados de [{st.session_state.get('yt_url', '?')}]({st.session_state.get('yt_url', '#')}) — "
        f"modelo{'s' if is_compare else ''}: **{models_label}**"
    )

    # KPIs con el modelo activo (o el único si no estamos en compare)
    kpi_model = active if active in model_keys else model_keys[0]
    verdict_col = f'_verdict_{kpi_model}'
    proba_col = f'_proba_{kpi_model}'

    toxic = int((df[verdict_col] == 'toxic').sum())
    nontoxic = int((df[verdict_col] == 'nontoxic').sum())
    uncertain = int((df[verdict_col] == 'uncertain').sum())
    total = len(df)

    if is_compare:
        st.caption(f"Las métricas se calculan con **{MODELS[kpi_model]['display']}**.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total comentarios", total)
    c2.metric("🔴 Tóxicos", f"{toxic} ({toxic / total * 100:.1f}%)")
    c3.metric("🟢 No tóxicos", nontoxic)
    c4.metric("🟡 Inciertos", uncertain)
    c5.metric("Prob. tóxico media", f"{df[proba_col].mean() * 100:.1f}%")

    # En modo compare, mostrar también cuántos discrepan
    if is_compare and len(model_keys) >= 2:
        a, b = model_keys[0], model_keys[1]
        disagreements = int((df[f'_pred_{a}'] != df[f'_pred_{b}']).sum())
        st.info(
            f"📊 Los dos modelos discrepan en **{disagreements}** comentarios "
            f"de {total} ({disagreements / total * 100:.1f}%)."
        )

    st.divider()

    # Filtro
    filter_options = ["Todos", "Solo tóxicos", "Solo no tóxicos", "Solo inciertos"]
    if is_compare and len(model_keys) >= 2:
        filter_options.append("Solo discrepancias")

    filtro = st.radio(
        f"Mostrar (filtro sobre {MODELS[kpi_model]['display']})" if is_compare else "Mostrar",
        filter_options,
        horizontal=True,
        key="yt_filter",
    )

    if filtro == "Solo tóxicos":
        df_show = df[df[verdict_col] == 'toxic']
    elif filtro == "Solo no tóxicos":
        df_show = df[df[verdict_col] == 'nontoxic']
    elif filtro == "Solo inciertos":
        df_show = df[df[verdict_col] == 'uncertain']
    elif filtro == "Solo discrepancias" and is_compare and len(model_keys) >= 2:
        a, b = model_keys[0], model_keys[1]
        df_show = df[df[f'_pred_{a}'] != df[f'_pred_{b}']]
    else:
        df_show = df

    # Columnas visibles
    base_cols = ['Comentario', 'Autor']
    for name in model_keys:
        display = MODELS[name]['display']
        base_cols += [f'Resultado ({display})', f'Prob. tóxico ({display})']
    base_cols += ['Likes', 'Fecha']

    styled = df_show.style.apply(color_by_verdict(verdict_col), axis=1)
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_order=base_cols,
        column_config={
            "Comentario": st.column_config.TextColumn(width="large"),
            "Autor": st.column_config.TextColumn(width="small"),
        },
    )

    csv = df_show[base_cols].to_csv(index=False)
    st.download_button(
        "⬇️ Descargar resultados CSV",
        csv,
        "resultados_toxicidad.csv",
        "text/csv",
    )

    if st.button("🗑️ Limpiar resultados", type="secondary"):
        for k in ('yt_results', 'yt_url', 'yt_models', 'yt_compare'):
            st.session_state.pop(k, None)
        st.rerun()

# --- Modo monitoreo en tiempo real ---
st.divider()
st.subheader("🔴 Monitoreo en tiempo real")
st.caption("Detecta comentarios nuevos automáticamente cada X segundos.")

col1, col2 = st.columns(2)
with col1:
    interval = st.slider("Intervalo de actualización (segundos)", 10, 60, 30)
with col2:
    monitor = st.toggle("Activar monitoreo", value=False)

if monitor:
    if not url_input.strip():
        st.warning("Introduce una URL antes de activar el monitoreo.")
    else:
        video_id = extract_video_id(url_input)
        if not video_id:
            st.error("URL no válida.")
        else:
            # Inicializar estado
            if 'monitor_seen' not in st.session_state:
                st.session_state['monitor_seen'] = set()
            if 'monitor_toxic' not in st.session_state:
                st.session_state['monitor_toxic'] = []

            status = st.empty()
            alert_box = st.empty()
            counter = st.empty()

            try:
                youtube = get_youtube()
                model, tfidf = get_model(active)

                with st.spinner("Obteniendo comentarios actuales..."):
                    comments = get_comments(youtube, video_id, 100)

                # Marcar comentarios existentes como ya vistos
                for c in comments:
                    st.session_state['monitor_seen'].add(c['text'][:50])

                status.success(
                    f"✅ Monitoreo activo — revisando cada {interval}s — "
                    f"{len(st.session_state['monitor_seen'])} comentarios base ignorados"
                )

                time.sleep(interval)

                # Nueva consulta
                new_comments = get_comments(youtube, video_id, 100)
                nuevos = [
                    c for c in new_comments
                    if c['text'][:50] not in st.session_state['monitor_seen']
                ]

                if nuevos:
                    for c in nuevos:
                        st.session_state['monitor_seen'].add(c['text'][:50])
                        text_clean = preprocess(c['text'])
                        pred, proba = predict(text_clean, model, tfidf)
                        verdict = label_for(proba)

                        if verdict == 'toxic':
                            st.session_state['monitor_toxic'].append({
                                'texto': c['text'],
                                'autor': c['author'],
                                'prob':  proba
                            })

                            if save_all:
                                try:
                                    save_prediction(
                                        text=c['text'],
                                        text_clean=text_clean,
                                        prediction=pred,
                                        probability=proba,
                                        source='realtime',
                                        video_url=url_input
                                    )
                                except Exception:
                                    pass

                    toxic_nuevos = [
                        c for c in nuevos
                        if label_for(predict(preprocess(c['text']), model, tfidf)[1]) == 'toxic'
                    ]

                    if toxic_nuevos:
                        alert_box.error(
                            f"🚨 {len(toxic_nuevos)} comentario(s) tóxico(s) nuevo(s) detectado(s)"
                        )
                    else:
                        alert_box.info(f"✅ {len(nuevos)} comentario(s) nuevo(s) — ninguno tóxico")
                else:
                    alert_box.info("No hay comentarios nuevos desde la última revisión.")

                # Mostrar historial de tóxicos detectados
                if st.session_state['monitor_toxic']:
                    counter.metric(
                        "🔴 Tóxicos detectados en esta sesión",
                        len(st.session_state['monitor_toxic'])
                    )
                    with st.expander("Ver comentarios tóxicos detectados"):
                        for item in st.session_state['monitor_toxic']:
                            st.error(
                                f"**{item['autor']}** ({item['prob']*100:.1f}%): {item['texto'][:200]}"
                            )

            except Exception as e:
                st.error(f"Error en monitoreo: {e}")

            # Rerun automático para continuar el monitoreo
            time.sleep(1)
            st.rerun()