"""
Genera notebooks/V4_eda/00_setup.ipynb con celdas pre-rellenadas.
Uso (desde la raíz del repo):
    uv run python notebooks/V4_eda/_make_00_setup.py
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# --- Celda 1: título ---
cells.append(nbf.v4.new_markdown_cell(
"""# 00 — Setup verification

**Objetivo:** verificar que todas las librerías que vamos a usar cargan sin errores y funcionan como esperamos.

Este notebook es el *hello world* del pipeline. Si alguna celda falla aquí, hay que arreglarla antes de seguir — no merece la pena programar lógica sobre cimientos rotos.
"""))

# --- Celda 2: introducción a versiones ---
cells.append(nbf.v4.new_markdown_cell(
"""## 1. Verificar versiones del stack

Importamos todas las librerías de golpe e imprimimos sus versiones. Si algo falta o está roto, el error sale en esta celda.
"""))

# --- Celda 3: código de versiones ---
cells.append(nbf.v4.new_code_cell(
"""import sys
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import sklearn
import nltk
import spacy
import wordcloud
import joblib

print(f"Python      : {sys.version.split()[0]}")
print(f"pandas      : {pd.__version__}")
print(f"numpy       : {np.__version__}")
print(f"matplotlib  : {matplotlib.__version__}")
print(f"seaborn     : {sns.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"NLTK        : {nltk.__version__}")
print(f"spaCy       : {spacy.__version__}")
print(f"wordcloud   : {wordcloud.__version__}")
print(f"joblib      : {joblib.__version__}")
"""))

# --- Celda 4: intro a NLTK ---
cells.append(nbf.v4.new_markdown_cell(
"""## 2. Sanity check de NLTK

Tokenizamos una frase y aplicamos la lista de stopwords en inglés.
Si funciona, los recursos `punkt` y `stopwords` están bien instalados.
"""))

# --- Celda 5: código NLTK ---
cells.append(nbf.v4.new_code_cell(
"""from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

frase = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(frase)
stop_en = set(stopwords.words('english'))
sin_stop = [t for t in tokens if t.lower() not in stop_en]

print(f"Original     : {frase}")
print(f"Tokens       : {tokens}")
print(f"Sin stopwords: {sin_stop}")
print(f"Stopwords EN : {len(stop_en)} palabras en total")
"""))

# --- Celda 6: intro a spaCy ---
cells.append(nbf.v4.new_markdown_cell(
"""## 3. Sanity check de spaCy

Cargamos el modelo `en_core_web_sm` y lematizamos una frase con palabras flexionadas. Fíjate cómo `were` → `be`, `running` → `run`, `children` → `child`.
"""))

# --- Celda 7: código spaCy ---
cells.append(nbf.v4.new_code_cell(
"""nlp = spacy.load('en_core_web_sm')

frase = "The children were running quickly through the woods."
doc = nlp(frase)

print(f"{'Token':<12} {'POS':<10} {'Lemma':<12}")
print("-" * 34)
for token in doc:
    print(f"{token.text:<12} {token.pos_:<10} {token.lemma_:<12}")
"""))

# --- Celda 8: intro a URL ---
cells.append(nbf.v4.new_markdown_cell(
"""## 4. Sanity check de la URL raw

Probamos el patrón fundamental del proyecto: cargar el dataset crudo directamente desde GitHub, sin paths locales. Si esto funciona, los notebooks siguientes pueden depender de esta misma URL sin problemas — y cualquier compañero/máquina/Colab podrá ejecutar el pipeline igual.

⚠️ Requiere internet.
"""))

# --- Celda 9: código URL ---
cells.append(nbf.v4.new_code_cell(
"""RAW_URL = "https://raw.githubusercontent.com/Bootcamp-IA-P6/P9E4/refs/heads/main/data/raw/youtoxic_english_1000.csv"

df = pd.read_csv(RAW_URL)

print(f"Shape: {df.shape}")
print(f"Columnas ({len(df.columns)}): {list(df.columns)}")
print()
df.head(3)
"""))

# --- Celda 10: conclusión ---
cells.append(nbf.v4.new_markdown_cell(
"""## Conclusión

Si las 4 celdas de código se han ejecutado sin error y los outputs tienen sentido, el setup está completo y validado.

**Próximo paso:** notebook `01_exploration.ipynb` — primer contacto serio con el dataset.
"""))

# Asignar celdas y metadata mínima
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3 (ipykernel)',
        'language': 'python',
        'name': 'python3',
    },
    'language_info': {'name': 'python', 'version': '3.14'},
}

# Escribir el fichero
output_path = os.path.join(os.path.dirname(__file__), '00_setup.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✅ Notebook creado: {output_path}")
print(f"   {len(cells)} celdas: {sum(1 for c in cells if c.cell_type == 'markdown')} markdown + {sum(1 for c in cells if c.cell_type == 'code')} code")