import streamlit as st


COLOR_TOXIC = "#FF6B6B"
COLOR_NONTOXIC = "#4ECDC4"
COLOR_UNCERTAIN = "#FFD166"


def page_header(title: str, subtitle: str = "", icon: str = ""):
<<<<<<< HEAD
    # Ocultar navegación automática de Streamlit
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {display: none;}
        </style>
    """, unsafe_allow_html=True)
    
=======
>>>>>>> 4d1cc9fa50a03ef6012b7c56a662e009e32e91ae
    """Cabecera consistente para todas las páginas."""
    icon_html = f"<span style='font-size: 2.2rem; margin-right: .6rem'>{icon}</span>" if icon else ""
    st.markdown(
        f"""
        <div style="padding: 1rem 0 .25rem 0">
            <div style="display:flex; align-items:center;">
                {icon_html}
                <h1 style="margin:0; font-weight:700; color:#E6E9EF;">{title}</h1>
            </div>
            {f'<p style="color:#9aa4b2; margin-top:.35rem; font-size:1.02rem;">{subtitle}</p>' if subtitle else ''}
        </div>
        <hr style="margin:.6rem 0 1.2rem 0; border:none; border-top:1px solid #2a2f3a;">
        """,
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, subtitle: str = ""):
    """Estado vacío centrado con icono grande."""
    st.markdown(
        f"""
        <div style="text-align:center; padding: 3rem 1rem; color:#9aa4b2;">
            <div style="font-size:4.5rem; line-height:1; margin-bottom:.6rem;">{icon}</div>
            <h3 style="color:#E6E9EF; margin:.3rem 0 .4rem 0;">{title}</h3>
            <p style="margin:0; font-size:.95rem;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def color_by_verdict(verdict_col: str):
    """Devuelve un styler de pandas que tinta cada fila según el veredicto."""
    def style_row(row):
        v = row[verdict_col]
        if v == "toxic":
            bg = "rgba(255, 107, 107, 0.16)"
        elif v == "nontoxic":
            bg = "rgba(78, 205, 196, 0.10)"
        elif v == "uncertain":
            bg = "rgba(255, 209, 102, 0.13)"
        else:
            bg = ""
        return [f"background-color: {bg}"] * len(row)
    return style_row


def render_video_preview(youtube, video_id: str):
    """Pinta thumbnail + título del vídeo. Devuelve dict con metadatos o None."""
    try:
        resp = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
        items = resp.get("items", [])
        if not items:
            return None
        info = items[0]["snippet"]
        stats = items[0].get("statistics", {})
        thumb = info["thumbnails"].get("medium", info["thumbnails"]["default"])["url"]
        title = info["title"]
        channel = info["channelTitle"]
        views = int(stats.get("viewCount", 0))
        # commentCount falta del payload cuando los comentarios están deshabilitados
        comments_n = stats.get("commentCount")
        comments_disabled = comments_n is None
        comments_n = int(comments_n) if comments_n is not None else 0

        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(thumb, use_container_width=True)
        with col2:
            st.markdown(f"### {title}")
            st.caption(f"📺 {channel}")
            mcol1, mcol2 = st.columns(2)
            mcol1.metric("👁 Visualizaciones", f"{views:,}".replace(",", "."))
            comments_label = "🚫 Deshabilitados" if comments_disabled else f"{comments_n:,}".replace(",", ".")
            mcol2.metric("💬 Comentarios totales", comments_label)

        return {
            "title": title,
            "channel": channel,
            "views": views,
            "comment_count": comments_n,
            "comments_disabled": comments_disabled,
        }
    except Exception:
        return None
