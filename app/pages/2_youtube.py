import streamlit as st
import sys
from pathlib import Path
import os
import re
import time
from googleapiclient.discovery import build
from dotenv import load_dotenv
import pandas as pd

load_dotenv(Path(__file__).parent.parent.parent / '.env')

sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import preprocess
from utils.model import load_model, predict
from utils.database import save_prediction

st.set_page_config(page_title="Análisis por URL", page_icon="🎥", layout="wide")

st.title("🎥 Análisis de comentarios por URL")
st.markdown("Introduce la URL de un vídeo de YouTube para analizar sus comentarios.")

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_youtube():
    api_key = os.getenv('YOUTUBE_API_KEY')
    return build('youtube', 'v3', developerKey=api_key)

model, tfidf = get_model()

def extract_video_id(url: str) -> str:
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
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
        textFormat='plainText'
    )
    while request and len(comments) < max_comments:
        response = request.execute()
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'text':   comment['textDisplay'],
                'author': comment['authorDisplayName'],
                'likes':  comment['likeCount'],
                'date':   comment['publishedAt'][:10]
            })
        request = youtube.commentThreads().list_next(request, response)
    return comments[:max_comments]

# --- UI ---
url_input = st.text_input(
    "URL del vídeo",
    placeholder="https://www.youtube.com/watch?v=..."
)

col1, col2 = st.columns(2)
with col1:
    max_comments = st.slider("Número máximo de comentarios", 10, 200, 50)
with col2:
    save_all = st.checkbox("Guardar predicciones en base de datos", value=True)

analyze = st.button("🔍 Analizar comentarios", type="primary")

if analyze and url_input.strip():
    video_id = extract_video_id(url_input)

    if not video_id:
        st.error("URL no válida. Asegúrate de que es un enlace de YouTube.")
    else:
        try:
            youtube = get_youtube()

            with st.spinner("Obteniendo comentarios..."):
                comments = get_comments(youtube, video_id, max_comments)

            if not comments:
                st.warning("No se encontraron comentarios en este vídeo.")
            else:
                st.info(f"Analizando {len(comments)} comentarios...")
                progress = st.progress(0)
                results = []

                for i, comment in enumerate(comments):
                    text_clean = preprocess(comment['text'])
                    pred, proba = predict(text_clean, model, tfidf)

                    results.append({
                        'Comentario': comment['text'][:100] + '...' if len(comment['text']) > 100 else comment['text'],
                        'Autor':      comment['author'],
                        'Resultado':  '🔴 Tóxico' if pred == 1 else '🟢 No tóxico',
                        'Probabilidad': f"{proba*100:.1f}%",
                        'Likes':      comment['likes'],
                        'Fecha':      comment['date'],
                        '_pred':      pred,
                        '_proba':     proba,
                        '_text':      comment['text'],
                        '_clean':     text_clean
                    })

                    if save_all:
                        try:
                            save_prediction(
                                text=comment['text'],
                                text_clean=text_clean,
                                prediction=pred,
                                probability=proba,
                                source='youtube',
                                video_url=url_input
                            )
                        except Exception:
                            pass

                    progress.progress((i + 1) / len(comments))

                progress.empty()

                # Métricas
                df = pd.DataFrame(results)
                toxic_count    = df['_pred'].sum()
                nontoxic_count = len(df) - toxic_count
                toxic_pct      = toxic_count / len(df) * 100

                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total comentarios", len(df))
                col2.metric("🔴 Tóxicos",     f"{toxic_count} ({toxic_pct:.1f}%)")
                col3.metric("🟢 No tóxicos",  f"{nontoxic_count}")
                col4.metric("Probabilidad media", f"{df['_proba'].mean()*100:.1f}%")

                st.divider()

                # Filtro
                filtro = st.radio(
                    "Mostrar",
                    ["Todos", "Solo tóxicos", "Solo no tóxicos"],
                    horizontal=True
                )

                if filtro == "Solo tóxicos":
                    df_show = df[df['_pred'] == 1]
                elif filtro == "Solo no tóxicos":
                    df_show = df[df['_pred'] == 0]
                else:
                    df_show = df

                st.dataframe(
                    df_show[['Comentario', 'Autor', 'Resultado', 'Probabilidad', 'Likes', 'Fecha']],
                    use_container_width=True,
                    hide_index=True
                )

                # Descargar CSV
                csv = df_show[['Comentario', 'Autor', 'Resultado', 'Probabilidad', 'Likes', 'Fecha']].to_csv(index=False)
                st.download_button(
                    "⬇️ Descargar resultados CSV",
                    csv,
                    "resultados_toxicidad.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"Error al obtener comentarios: {e}")

elif analyze and not url_input.strip():
    st.warning("Por favor introduce una URL.")