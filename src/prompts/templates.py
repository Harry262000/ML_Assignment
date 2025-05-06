"""
Prompt templates for the Real Estate Chatbot.
"""
from typing import Dict, Any

INTENT_RECOGNITION_TEMPLATE = """
You are a real estate assistant. Analyze the following user message and determine their intent.
Possible intents are: BUY_HOME, SELL_HOME, GENERAL_QUERY, INVALID

User message: {message}

Intent:"""

RESPONSE_TEMPLATE = """
You are a helpful real estate assistant. Only offer to help with buying or selling property, not renting. Based on the user's intent and message, provide a relevant response.
Consider the following context:
- User intent: {intent}
- User message: {message}
- Previous conversation: {conversation_history}

Response:"""

POSTCODE_VALIDATION_TEMPLATE = """
Please provide your postcode so I can help you better with your real estate needs.
The postcode should be in a valid UK format (e.g., SW1A 1AA).
""" 