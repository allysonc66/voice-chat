from langdetect import detect
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pyttsx3
from googletrans import Translator
import os
import google.generativeai as genai

class VoiceChatBot:
    def __init__(self, model_name="gemini-1.5-flash"):
        """Initialize the VoiceChatBot with model and settings."""
        self.translator = Translator()
        self.access_token = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=self.access_token)
        self.model = genai.GenerativeModel(model_name=model_name)
        self.chat = self.model.start_chat()

        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.rate = self.tts_engine.getProperty('rate')
        self.tts_engine.setProperty('rate', self.rate - 50)

        # Voice settings for different languages
        self.voice_lang = {
            "en-US": "com.apple.voice.compact.en-US.Samantha",
            "es-ES": "com.apple.eloquence.es-ES.Rocko",
            "fr-FR": "com.apple.voice.compact.fr-CA.Amelie"
        }

        # Starter messages for each language and topic
        self.starter_messages = {
            "en-US": {
                "context": "The following is a casual conversation between 2 people that speak Ensligh. ",
                "family": "How many people are in your family?",
                "food": "What is your favorite food?",
                "animals": "Do you have any pets?",
                "travel": "Where do you want to travel?",
                "hobbies": "What are your hobbies?"
            },
            "es-ES": {
                "context": "La siguientes es una conversación casual entre 2 personas que hablan Espanol. ",
                "family": "¿Cuántas personas hay en tu familia?",
                "food": "¿Cuál es tu comida favorita?",
                "animals": "¿Tienes algún animal?",
                "travel": "¿A dónde quieres viajar?",
                "hobbies": "¿Cuáles son tus pasatiempos?"
            },
            "fr-FR": {
                "context": "La suivante est une conversation de 2 personnes qui parlent francophone. ",
                "family": "Combien de personnes sont dans votre famille?",
                "food": "Quel est votre plat préféré?",
                "animals": "Avez-vous des animaux?",
                "travel": "Où voulez-vous voyager?",
                "hobbies": "Quels sont vos loisirs?"
            }
        }

    def detect_language(self, text):
        """Detect the language of the given text."""
        return detect(text)

    def generate_response(self, text, rewind_context=False):
        """Generate a response based on the input text and conversation context."""
        if rewind_context:
            self.chat.rewind()
        response = self.chat.send_message(text)
        return response.text

    def text_to_speech(self, text, language):
        """Convert text to speech in the specified language."""
        if language not in self.voice_lang:
            raise ValueError(f"Unsupported language: {language}")
        if self.tts_engine._inLoop:
            self.tts_engine.endLoop()
        self.tts_engine.setProperty('voice', self.voice_lang[language])
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def get_conversation_starter(self, topic, language):
        """Get a conversation starter based on the topic and language."""
        context = self.starter_messages[language]["context"]
        start_message = self.starter_messages[language][topic]
        self.chat = self.model.start_chat(history=[{"role": "model", "parts": context + start_message}])

        return start_message

    def translate_text(self, text, from_lang="es", to_lang="en"):
        """Translate text from one language to another."""
        return self.translator.translate(text, src=from_lang, dest=to_lang).text



app = Flask(__name__)
CORS(app)

# Create an instance of VoiceChatBot
chatbot = VoiceChatBot()

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

    starter = chatbot.get_conversation_starter(topic, language)
    chatbot.text_to_speech(starter, language)

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

    translated_text = chatbot.translate_text(text, from_lang, to_lang)
    return jsonify({'translation': translated_text}), 200

@app.route('/chatbot', methods=['POST'])
def chatbot_endpoint():
    """Handle chatbot interactions."""
    data = request.get_json()
    text = data.get('text')
    rewind_context = data.get('rewindContext', False)
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    language = data.get('language', "en-US")
    response = chatbot.generate_response(text, rewind_context)
    chatbot.text_to_speech(response, language)

    return jsonify({"response": response}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5001)