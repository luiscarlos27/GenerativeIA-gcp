__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import ollama
import chromadb

documents = [
  "El IPC desciende hasta el 3,2% en marzo en Guatemala 2024"
  "La tasa de variación anual del IPC en Guatemala en marzo de 2024 ha sido del 3,2%,",
  "1 décima inferior a la del mes anterior. La variación mensual del IPC (Índice de Precios al Consumo) ha sido del 0,3%,",
  "de forma que la inflación acumulada en 2024 es del 0,5%.",
  "Hay que destacar la subida del 2% de los precios de Transporte,",
  "hasta situarse su tasa interanual en el 1,2%,",
  "que contrasta con el descenso de los precios de Otros bienes y servicios del -0,1%,",
  "y una variación interanual del 2,4%."
]

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
  prompt = "Necesito informacion sobre el IPC de Guatemala del 2024?"

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
    prompt=f"Usando estos datos: {data}. Responde este prompt: {prompt}"
  )

  print(output['response'])

