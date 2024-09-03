import unittest
from app import VoiceChatBot

class TestChatBot(unittest.TestCase):
    def setUp(self):
        self.chatbot = VoiceChatBot('The widget')

    def test_translate_text(self):
        resp = self.chatbot.translate_text("La siguientes es una conversación casual entre 2 personas que hablan Espanol.", "es", "en")
        self.assertEqual(resp, "The following is a casual conversation between 2 people who speak Spanish.")

    
    def test_detect_language(self):
        resp = self.chatbot.detect_language("La siguientes es una conversación casual entre 2 personas que hablan Espanol.")
        self.assertEqual(resp, 'es')
    
    def test_starter_messages(self):
        resp = self.chatbot.get_conversation_starter("family", "en-US")
        self.assertEqual(resp, 'How many people are in your family?')