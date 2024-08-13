from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BlenderbotTokenizer, BlenderbotForConditionalGeneration
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pyttsx3
from googletrans import Translator
import torch
import os

app = Flask(__name__)
CORS(app)

access_token = os.environ.get("ACCESS_TOKEN")

model_name = "meta-llama/Meta-Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Store the conversation context
conversation_context = {}

tts_engine = pyttsx3.init()

rate = tts_engine.getProperty('rate')
tts_engine.setProperty('rate', rate-50)

voice_lang = {
    "en-US": "com.apple.voice.compact.en-US.Samantha",
    "es-ES": "com.apple.eloquence.es-ES.Rocko",
    "fr-FR": "com.apple.voice.compact.fr-CA.Amelie"
}

language_string = {
    "en-US": "English",
    "es-ES": "Spanish",
    "fr-FR": "French"
}

# Starter message for each language and each topic
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

translator = Translator()


def detect_language(text):
    language = detect(text)
    print(f"Detected Language: {language}")
    return language

def generate_response(text, language, session_id):
    conversation = conversation_context[session_id]
    conversation += '''
    Miguel: {text}. 
    Alicia: 
    '''.format(text=text, end_token=tokenizer.eos_token, prompt=starter_messages[language]['prompt'])

    print("CONVERSATION", conversation)

    # Generate the response
    input_ids = tokenizer.encode(conversation, return_tensors='pt')
    input_ids = input_ids.to(device)

    print("HERE")
    outputs = model.generate(input_ids, max_length=100, repetition_penalty=1.2, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[-1], skip_special_tokens=True)
    print("OUTPUT", response)
    
    conversation += response.split("Alicia:")[-1]
    conversation_context[session_id] = conversation

    return response


def text_to_speech(text, language):
    if tts_engine._inLoop:
        tts_engine.endLoop()
    tts_engine.setProperty('voice', voice_lang[language])
    tts_engine.say(text)
    tts_engine.runAndWait()


def get_conversation_starter(topic, language, session_id):
    prompt = starter_messages[language]['prompt']
    response = '''
    {prompt}

    Alicia: {topic}.
    '''.format(prompt=prompt, topic=starter_messages[language][topic], end_token=tokenizer.eos_token)
    conversation_context[session_id] = response

    return starter_messages[language][topic]


def translate_text(text, from_lang="es", to_lang="en"):
    translated = translator.translate(text, src=from_lang, dest=to_lang).text
    print("translation", translated)
    return translated


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat')
def chat():
    return render_template('chat.html')


@app.route('/start_conversation', methods=['POST'])
def start_conversation():
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
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']
    session_id = request.remote_addr
    if text == "":
        return jsonify({'error': 'No text provided'}), 400
    language = data['language'] if 'language' in data else "en-US"
    response = generate_response(text, language, session_id)
    # response = "¿Hola, como puedo ayudarte?"
    text_to_speech(response, language)
    return jsonify({"response": response}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
    # TODO: (3) Center the mic icon and make it more obvious - should be red on stop
    # TODO: (5) Make topic selection look nicer (should be centered)
    # TODO: (8) Add a "quick translate" side bar that allows you to quickly look up words
    # TODO: (9) Add pronunciation feedback
    # TODO: (10) Deploy to AWS
