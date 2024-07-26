__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import ollama
import chromadb
import csv

# Read recorrido.csv
with open('consumo.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)

    valores = [str(row) for row in reader]
    documents = valores[::-1]
    # Prepare data for Ollama API (if available)

print(documents)
client = chromadb.Client()
collection = client.create_collection(name="docs")

# store each document in a vector embedding database
for i, d in enumerate(documents):
  response = ollama.embeddings(model="phi3:mini", prompt=d)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[d]
  )

  # an example prompt
  prompt = "Como un analista se requiere analizar los de consumo de combustible la estructura del informe en formato csv se detalle a continuación: la primera columna con el nombre 'Fecha' hace referencia a la fecha y hora de la lectura del sensor, la columna 2 con el nombre 'Sensor' hace referencia al nombre del sensor, la columna con el nombre 'Volumen (Gals)' hace referencia a la cantidad de galones que tiene el tanque al momento de la medición, la columna numero 4 con el nombre 'Distancia (KM)' hace referencia a la distancia recorrida desde el primer registro analizado, la quinta columna con el nombre 'Ver Mapa' es solo de referencia no tomarlo en consideración. Se necesita brindar un breve resumen sobre el consumo de combustible y determinar si existe alguna anomalia con el consumo y si existe alguna posible extracción basado en el consumo promedio de la unidad - NO MOSTRAR LOS DATOS DE MEDICION UNO A UNO SOLO EL RESUMEN CON LOS EVENTOS HORA Y FECHA DE LOS POSIBLES EXTRACCIONES - UNA DIFERENCIA DE 4 GALONES EN PROMEDIO POR HORA ES UNA POSIBLE EXTRACCION."

  # generate an embedding for the prompt and retrieve the most relevant doc
  response = ollama.embeddings(
    prompt=prompt,
    model="phi3:mini"
  )
  results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
  )
  data = results['documents'][0][0]

  # generate a response combining the prompt and data we retrieved in step 2
  output = ollama.generate(
    model="phi3:mini",
    prompt=f"Usando la siguiente data: {data}. Responde este enunciado: {prompt}"
  )

  print(output['response'])