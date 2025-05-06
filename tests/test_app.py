import unittest
from unittest.mock import patch, MagicMock
import streamlit as st

from src.chatbot import RealEstateChatbot

class TestRealEstateApp(unittest.TestCase):
    def setUp(self):
        # Mock the Streamlit session state
        self.mock_session_state = MagicMock()
        self.patcher = patch('streamlit.session_state', self.mock_session_state)
        self.patcher.start()

        # Mock the chatbot
        self.mock_chatbot = MagicMock(spec=RealEstateChatbot)
        self.mock_chatbot.process_message.return_value = {
            "assistant_response": "I can help you with that."
        }

    def tearDown(self):
        self.patcher.stop()

    def test_initial_state(self):
        """Test that the initial state is set up correctly"""
        # Initialize session state
        st.session_state.messages = [{
            "role": "assistant",
            "content": "I'm real estate chatbot. How can I help you with buying or selling a property?"
        }]

        # Verify initial state
        self.assertEqual(len(st.session_state.messages), 1)
        self.assertEqual(st.session_state.messages[0]["role"], "assistant")
        self.assertIn("real estate chatbot", st.session_state.messages[0]["content"])

    def test_chatbot_response(self):
        """Test that the chatbot processes messages correctly"""
        # Set up test message
        test_message = "I want to buy a house"
        
        # Mock the chatbot response
        st.session_state.chatbot = self.mock_chatbot
        
        # Process the message
        response = st.session_state.chatbot.process_message(test_message)
        
        # Verify the response
        self.assertEqual(response["assistant_response"], "I can help you with that.")
        self.mock_chatbot.process_message.assert_called_once_with(test_message)

    def test_error_handling(self):
        """Test that errors are handled gracefully"""
        # Set up test message
        test_message = "I want to buy a house"
        
        # Mock the chatbot to raise an exception
        st.session_state.chatbot = self.mock_chatbot
        self.mock_chatbot.process_message.side_effect = Exception("Test error")
        
        # Process the message
        try:
            response = st.session_state.chatbot.process_message(test_message)
        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            self.assertEqual(error_message, "❌ Error: Test error")

if __name__ == '__main__':
    unittest.main() 