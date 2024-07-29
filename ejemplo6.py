__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import ollama
import chromadb
import csv
import pandas as pd

# Leer el archivo CSV y convertirlo en un DataFrame
df = pd.read_csv('consumo.csv')

# Dividir el DataFrame en trozos más pequeños (ajustar según el tamaño deseado)
chunksize = 500
chunks = [df[i:i+chunksize] for i in range(0, len(df), chunksize)]

# Crear una colección en ChromaDB
client = chromadb.Client()
collection = client.create_collection(name="docs")

# Procesar cada chunk
for i, chunk in enumerate(chunks):
    # Convertir el chunk en una lista de diccionarios
    data = chunk.to_dict(orient='records')

    # Crear un prompt más específico
    prompt = f"""
    Analiza los siguientes datos de consumo de combustible: {data}. 
    Identifica posibles anomalías en el consumo, como:
    * Aumentos o disminuciones significativas en el consumo por día.
    * Discrepancias entre el consumo y la distancia recorrida.
    * Patrones inusuales en los horarios de consumo.
    * Valores atípicos en el volumen de combustible.

    Proporciona un resumen conciso de tus hallazgos, incluyendo fechas y horas de las posibles anomalías.
    """

    # Obtener la embedding del prompt
    response = ollama.embeddings(
        prompt=prompt,
        model="phi3:mini"
    )
    embedding = response["embedding"]

    # Agregar el chunk y su embedding a ChromaDB con IDs únicos
    collection.add(
        ids=[f"chunk_{i}"],  # Crear IDs únicos para cada chunk
        embeddings=[embedding],
        documents=[data]
    )

    # Realizar una búsqueda similar para encontrar el documento más relevante
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1
    )
    data = results['documents'][0][0]

    # Generar una respuesta usando el modelo Ollama
    output = ollama.generate(
        model="phi3:mini",
        prompt=f"Usando la siguiente data: {data}. Responde este enunciado: {prompt}"
    )

    print(output['response'])