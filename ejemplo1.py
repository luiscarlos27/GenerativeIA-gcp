__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import ollama
import chromadb

documents = [
  "La sangre de las llamas c"
  "ontiene una gran cantidad de hemoglobina, la proteína que transporta el oxígeno de los pulmones a todo el cuerpo, lo que les permite sobrevivir a gran altura con bajos niveles de oxígeno.",
  "Las llamas son rumiantes modificados con estómagos de tres cámaras que les permiten procesar una variedad de follaje en su entorno hostil",
  "Las llamas tienen piel gruesa para mantenerlos calientes y ayudar a protegerlos contra las mordeduras de animales",
  "Una velocidad de carrera máxima de 40 millas por hora (65 km/h) les ayuda a escapar de los depredadores.",
  "Las hembras son más pequeñas que los machos.",
  "Tienen cuellos y patas largos, cabezas relativamente pequeñas con el labio superior partido, orejas grandes y colas cortas.",
  "Sus pies tienen dos dedos con uñas duras y una almohadilla correosa en la planta del dedo."
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
    documents=[d]locals()
  )

  # an example prompt
  prompt = "Brindame informacion sobre las llamas y su morfoligia?"

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
    prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
  )

  print(output['response'])

