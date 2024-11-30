import openai
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

# Configurar la API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")  # Asegúrate de configurar la variable de entorno

# Inicializar la aplicación Flask
app = Flask(__name__)

# Función para procesar el texto y generar la recomendación
def procesar_texto(prompt_texto):
    # Llamar a la API de OpenAI con parámetros para limitar la longitud
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un asistente que analiza emociones en texto y ofrece recomendaciones empáticas muy breves y concretas para ayudar en situaciones difíciles."},
            {"role": "user", "content": f"Texto: {prompt_texto}"}
        ],
        # Configuraciones para limitar la generación de tokens
        max_tokens=300,  # Limita la respuesta a aproximadamente 300 tokens
        temperature=0.7,  # Mantiene cierta creatividad pero con más control
        top_p=0.9,  # Controla la diversidad de la generación
        frequency_penalty=0.5,  # Reduce la repetición
        presence_penalty=0.5  # Evita respuestas demasiado similares
    )
    # Retornar la recomendación generada
    return response.choices[0].message.content

# Ruta de la API para procesar el texto
@app.route('/procesar-texto', methods=['POST'])
def analizar_texto():
    # Verificar que el cuerpo de la solicitud contenga el texto
    data = request.get_json()
    if 'prompt_texto' not in data:
        return jsonify({"error": "El campo 'prompt_texto' es requerido"}), 400

    # Obtener el texto de la solicitud
    prompt_texto = data['prompt_texto']

    # Generar la recomendación
    resultado = procesar_texto(prompt_texto)

    # Retornar la recomendación como JSON
    return jsonify({"resultado": resultado})

# Ejecutar la aplicación en el puerto 5000
if __name__ == '__main__':
    app.run(debug=True, port=5000)