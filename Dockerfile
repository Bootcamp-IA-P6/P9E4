FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv
RUN pip install uv

# Copiar archivos de dependencias
COPY pyproject.toml .
COPY uv.lock* .

# Instalar dependencias
RUN uv sync --frozen --no-dev

# Copiar el proyecto
COPY . .

# Descargar recursos NLTK
RUN uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('omw-1.4')"

# Exponer puerto de Streamlit
EXPOSE 8501

# Variables de entorno para Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Ejecutar la app
CMD ["uv", "run", "streamlit", "run", "app/app.py"]