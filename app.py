from langdetect import detect
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pyttsx3
from googletrans import Translator
import torch
import os

app = Flask(__name__)
CORS(app)

# Load environment variables
access_token = os.environ.get("ACCESS_TOKEN")

# Initialize model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Store the conversation context
conversation_context = {}

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
rate = tts_engine.getProperty('rate')
tts_engine.setProperty('rate', rate - 50)

# Voice settings for different languages
voice_lang = {
    "en-US": "com.apple.voice.compact.en-US.Samantha",
    "es-ES": "com.apple.eloquence.es-ES.Rocko",
    "fr-FR": "com.apple.voice.compact.fr-CA.Amelie"
}

# Starter messages for each language and topic
starter_messages = {
    "en-US": {
        "prompt": "system: An informal conversation in Spanish between Alicia and Miguel. Alice responds to your statement or question and then asks a follow-up question. Alicia never answers more than 1 statement.",
        "family": "How many people are in your family?",
        "food": "What is your favorite food?",
        "animals": "Do you have any pets?",
        "travel": "Where do you want to travel?",
        "hobbies": "What are your hobbies?"
    },
    "es-ES": {
        "prompt": "sistema: Una conversación informal en español entre Alicia y Miguel. Alicia responde a su afirmación o pregunta y luego hace una pregunta de seguimiento. Alicia nunca responde más de 1 declaracion.",
        "family": "¿Cuántas personas hay en tu familia?",
        "food": "¿Cuál es tu comida favorita?",
        "animals": "¿Tienes algún animal?",
        "travel": "¿A dónde quieres viajar?",
        "hobbies": "¿Cuáles son tus pasatiempos?"
    },
    "fr-FR": {
        "prompt": "système : Une conversation informelle en espagnol entre Alicia et Miguel. Alice répond à votre déclaration ou question, puis pose une question complémentaire. Alicia ne répond jamais à plus de 1 affirmation.",
        "family": "Combien de personnes sont dans votre famille?",
        "food": "Quel est votre plat préféré?",
        "animals": "Avez-vous des animaux?",
        "travel": "Où voulez-vous voyager?",
        "hobbies": "Quels sont vos loisirs?"
    }
}

# Initialize translator
translator = Translator()

def detect_language(text):
    """Detect the language of the given text."""
    return detect(text)

def generate_response(text, language, session_id):
    """Generate a response based on the input text and conversation context."""
    conversation = conversation_context.get(session_id, "")
    conversation += f"\nMiguel: {text}.\nAlicia: "
    
    # Generate the response
    input_ids = tokenizer.encode(conversation, return_tensors='pt').to(device)
    outputs = model.generate(input_ids, max_length=100, repetition_penalty=1.2, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    conversation += response.split("Alicia:")[-1]
    conversation_context[session_id] = conversation

    return response

def text_to_speech(text, language):
    """Convert text to speech."""
    if language not in voice_lang:
        raise ValueError(f"Unsupported language: {language}")
    if tts_engine._inLoop:
        tts_engine.endLoop()
    tts_engine.setProperty('voice', voice_lang[language])
    tts_engine.say(text)
    tts_engine.runAndWait()

def get_conversation_starter(topic, language, session_id):
    """Get a conversation starter based on the topic and language."""
    prompt = starter_messages[language]['prompt']
    response = f"{prompt}\n\nAlicia: {starter_messages[language][topic]}."
    conversation_context[session_id] = response

    return starter_messages[language][topic]

def translate_text(text, from_lang="es", to_lang="en"):
    """Translate text from one language to another."""
    return translator.translate(text, src=from_lang, dest=to_lang).text

@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/chat')
def chat():
    """Render the chat page."""
    return render_template('chat.html')

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    """Start a new conversation based on the provided topic and language."""
    data = request.get_json()
    topic = data.get('topic')
    language = data.get('language')
    if not topic or not language:
        return jsonify({'error': 'No topic or language provided'}), 400

    session_id = request.remote_addr
    starter = get_conversation_starter(topic, language, session_id)
    text_to_speech(starter, language)

    return jsonify({'starter': starter}), 200

@app.route('/translate', methods=['POST'])
def translate_endpoint():
    """Translate text from one language to another."""
    data = request.get_json()
    text = data.get('text')
    from_lang = data.get('from_lang')
    to_lang = data.get('to_lang')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    translated_text = translate_text(text, from_lang, to_lang)
    return jsonify({'translation': translated_text}), 200

@app.route('/chatbot', methods=['POST'])
def chatbot_endpoint():
    """Handle chatbot interactions."""
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    session_id = request.remote_addr
    language = data.get('language', "en-US")
    response = generate_response(text, language, session_id)
    text_to_speech(response, language)

    return jsonify({"response": response}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)