import speech_recognition as sr
from langdetect import detect
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pyttsx3
from googletrans import Translator

app = Flask(__name__)
CORS(app)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

recognizer = sr.Recognizer()

tts_engine = pyttsx3.init()

rate = tts_engine.getProperty('rate')
tts_engine.setProperty('rate', rate-50)

voice_lang = {
    "en-US": "com.apple.voice.compact.en-US.Samantha",
    "es-ES": "com.apple.eloquence.es-ES.Rocko",
    "fr-FR": "com.apple.voice.compact.fr-CA.Amelie"
}

translator = Translator()


def recognize_speech(audio_file):
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print(
                f"Could not request results from Google Speech Recognition service; {e}")


def detect_language(text):
    language = detect(text)
    print(f"Detected Language: {language}")
    return language


def generate_response(text):
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Response: {response}")
    return response


def text_to_speech(text, language):
    if tts_engine._inLoop:
        tts_engine.endLoop()
    tts_engine.setProperty('voice', voice_lang[language])
    tts_engine.say(text)
    tts_engine.runAndWait()


def get_conversation_starter(topic, language):
    starters = {
        "travel": {
            "en-US": "Where is your favorite place to travel?",
            "es-ES": "¿Cuál es tu lugar favorito para viajar?",
            "fr-FR": "Quel est votre endroit préféré pour voyager?"
        },
        "food": {
            "en-US": "What is your favorite food?",
            "es-ES": "¿Cuál es tu comida favorita?",
            "fr-FR": "Quel est votre plat préféré?"
        },
        "family": {
            "en-US": "Tell me about your family.",
            "es-ES": "Háblame de tu familia.",
            "fr-FR": "Parlez-moi de votre famille."
        },
        "hobbies": {
            "en-US": "What are your hobbies?",
            "es-ES": "¿Cuáles son tus pasatiempos?",
            "fr-FR": "Quels sont vos passe-temps?"
        }
    }
    return starters.get(topic, {}).get(language, "Hello, how can I help you?")


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

    starter = get_conversation_starter(topic, language)
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
    language = data['language'] if 'language' in data else "en-US"
    # response = generate_response(text)
    response = "¿Hola, como puedo ayudarte?"
    text_to_speech(response, language)
    return jsonify({"response": response}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
    # TODO: (3) Center the mic icon and make it more obvious - should be red on stop
    # TODO: (5) Make topic selection look nicer (should be centered)
    # TODO: (8) Add a "quick translate" side bar that allows you to quickly look up words
    # TODO: (9) Add pronunciation feedback
    # TODO: (10) Deploy to AWS
