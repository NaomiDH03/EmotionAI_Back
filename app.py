import openai
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from PIL import Image
import numpy as np
from fer import FER
import io


# Cargamos variables de enotrno
load_dotenv()

# Esto es para la api que ya se encuentra en mi .env
openai.api_key = os.getenv("OPENAI_API_KEY")

# Inicializar la aplicación Flask
app = Flask(__name__)

# Ponemos esto ya que tuve problemillas
CORS(app, origins=["http://localhost:5173"])


# Con esta función analizamos el texto
def analizar_texto(prompt_texto):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un asistente diseñado para analizar textos y proporcionar interpretaciones emocionales. Solo responde con la emocion dominante que detectes"},
            {"role": "user", "content": f"Análisis de texto: {prompt_texto}"}
        ],
        max_tokens=15,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
    return response.choices[0].message.content

# Con esta función analizamos la imagen
def analizar_imagen(prompt_imagen):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un asistente diseñado para analizar imágenes y proporcionar interpretaciones emocionales. Eres un asistente diseñado para analizar imágenes y proporcionar interpretaciones emocionales. Solo responde con una sola palabra que represente la emoción dominante detectada, sin añadir texto adicional."},
            {"role": "user", "content": f"Análisis de imagen: {prompt_imagen}"}
        ],
        max_tokens=15,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
    return response.choices[0].message.content


# Con esta funcion procesamos nuestra imagen y obetenemos la emocion 
def obtener_emocion_imagen(file):
    # Convertir el archivo a una imagen usando PIL
    img = Image.open(io.BytesIO(file.read()))

    # Lo convertimos a un array de numpy porque asi lo espera FER
    img_np = np.array(img)

    # Usamos fer
    detector = FER()
    emociones = detector.top_emotion(img_np) 
    
    # Retornamos la emocion
    return emociones


# Ruta para analizar solo la imagen y predecir la emoción
@app.route('/analizar-imagen', methods=['POST'])
def analizar_imagen_endpoint():
    global resultado_imagen
    if 'prompt_imagen' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400
    
    file = request.files['prompt_imagen']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    # Obtener la emoción de la imagen
    emocion = obtener_emocion_imagen(file)
    
    # Generar el prompt para OpenAI basado en la emoción
    prompt_imagen = f"La persona en la imagen parece estar {emocion[0]}. ¿Qué emoción transmite esta imagen?"

    # Analizar la imagen usando la API de OpenAI
    resultado_imagen = analizar_imagen(prompt_imagen)
    
    return jsonify({'resultado_imagen': resultado_imagen})

@app.route('/analizar-texto', methods=['POST'])
def analizar_texto_endpoint():
    global resultado_texto
    data = request.get_json()
    if 'prompt_texto' not in data:
        return jsonify({"error": "El campo 'prompt_texto' es requerido"}), 400

    prompt_texto = data['prompt_texto']
    resultado_texto = analizar_texto(prompt_texto)
    return jsonify({"resultado_texto": resultado_texto})

@app.route('/analizar-combinado', methods=['POST'])
def analizar_combinado_endpoint():
    if 'prompt_imagen' not in request.files or 'prompt_texto' not in request.form:
        return jsonify({'error': 'Se requiere tanto una imagen como un texto'}), 400
    
    file = request.files['prompt_imagen']
    prompt_texto = request.form['prompt_texto']
    
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    

    emocion = obtener_emocion_imagen(file)

    prompt_imagen = f"La persona en la imagen parece estar {emocion[0]}. ¿Qué emoción transmite esta imagen?"
    

    resultado_imagen = analizar_imagen(prompt_imagen)
    

    resultado_texto = analizar_texto(prompt_texto)
    
    # Aqui genramos la respuesta combianda
    sugerencia_prompt = f"Texto analizado: {resultado_texto}. Imagen analizada: {resultado_imagen}. ¿Qué sugerencia puedes dar basada en estos análisis?"
    sugerencia = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un asistente que da sugerencias empáticas basadas en análisis de texto e imagen, ofreciendo respuestas empáticas y adaptadas a las posibles diferencias emocionales, se breve"},
            {"role": "user", "content": sugerencia_prompt}
        ],
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5
    ).choices[0].message.content
    
    return jsonify({'sugerencia': sugerencia})  # Nos devuelve su recomendación/sugerencia :)



@app.route('/')
def home():
    return "DESPLEGADO"

# Ejecutar la aplicación
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
