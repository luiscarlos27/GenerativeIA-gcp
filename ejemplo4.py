import csv
import requests
import json

# Replace with the actual URL and API key for your Ollama instance
API_URL = "http://localhost:11434/api/generate"
API_KEY = "3241234123dfasdfad"

def analyze_recorrido_csv(filepath):
    """
    Reads recorrido.csv, sends data to Ollama API (if available), and provides a summary.

    Args:
        filepath (str): Path to the recorrido.csv file.

    Returns:
        str: A human-readable summary of the data (or error message).
    """

    try:
        # Read recorrido.csv
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            valores = [row for row in reader]
            # Prepare data for Ollama API (if available)
            data = {
                "model": "phi3:mini",
                "prompt": "Como un analista se requiere analizar los de consumo de combustible la estructura del informe en formato csv se detalle a continuación: la primera columna con el nombre 'Fecha' hace referencia a la fecha y hora de la lectura del sensor, la columna 2 con el nombre 'Sensor' hace referencia al nombre del sensor, la columna con el nombre 'Volumen (Gals)' hace referencia a la cantidad de galones que tiene el tanque al momento de la medición, la columna numero 4 con el nombre 'Distancia (KM)' hace referencia a la distancia recorrida desde el primer registro analizado, la quinta columna con el nombre 'Ver Mapa' es solo de referencia no tomarlo en consideración. Se necesita brindar un breve resumen sobre el consumo de combustible y determinar si existe alguna anomalia con el consumo y si existe alguna posible extracción basado en el consumo promedio de la unidad - NO MOSTRAR LOS DATOS DE MEDICION UNO A UNO SOLO EL RESUMEN. Los datos son los que a continuación se detallan:  \n".join(map(str, valores))
            }

            #print(data)

            if API_KEY:  # Use Ollama if API key is provided
                #headers = {"Authorization": f"Bearer {API_KEY}"}
                response = requests.post(API_URL, json=data, stream=True)
                for line in response.iter_lines():
                    if line:
                        json_data = json.loads(line)
                        if json_data['done'] == False:
                            print(json_data['response'], end='', flush=True)

    except FileNotFoundError:
        return "Error: recorrido.csv file not found."
    except Exception as e:
        return f"Error: {e}"

# Example usage (replace with your actual file path)
filepath = "consumo.csv"
summary = analyze_recorrido_csv(filepath)
print(summary)