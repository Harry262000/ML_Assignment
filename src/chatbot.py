from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from openai import OpenAI
from src.memory.vector_store import VectorStore
from src.prompts.templates import (
    INTENT_RECOGNITION_TEMPLATE,
    RESPONSE_TEMPLATE,
    SLOT_PROMPT_TEMPLATE,
)
import json
import os
from src.utils.postcode_validator import validate_postcode, load_uk_postcodes, validate_uk_postcode_format

# from src.prompts.templates import INTENT_RECOGNITION_TEMPLATE, RESPONSE_TEMPLATE
# from src.agent.slot_filling import run_slot_filling, SLOTS

# ── Slot‑filling setup ──────────────────────────────────────────────────────
SLOT_KEYS = ["intent", "property_type", "name", "phone", "email", "budget", "postcode"]


class RealEstateChatbot:
    def __init__(self, api_key: str, vector_store: Optional[VectorStore] = None):
        """
        Initialize the chatbot.

        Args:
            api_key: OpenAI API key
            vector_store: Optional vector store for conversation memory (defaults to in-memory storage)
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        # If no vector store is provided, use the in-memory vector store
        # self.vector_store = vector_store or VectorStore()  # This will use in-memory by default
        self.vector_store = vector_store  # This will use in-memory by default
        self.conversation_history = []
        # slot state persists across turns until filled
        self.slot_state: Dict[str, Any] = {k: None for k in SLOT_KEYS}

        self.model = "openai/gpt-4o"

        # Load valid postcodes
        try:
            # Get the absolute path to the project root directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            postcode_file = os.path.join(project_root, 'data', 'uk_postcodes.csv')
            
            if not os.path.exists(postcode_file):
                print(f"Warning: Postcode file not found at {postcode_file}")
                self.valid_postcodes = []
            else:
                self.valid_postcodes = load_uk_postcodes(postcode_file)
        except Exception as e:
            print(f"Error loading postcodes: {e}")
            self.valid_postcodes = []

    def validate_user_postcode(self, postcode: str) -> Tuple[bool, str, str]:
        """
        Validate if the user's postcode is in our service area.
        
        Args:
            postcode: User's postcode to validate
            
        Returns:
            Tuple of (is_valid, formatted_postcode, area)
        """
        return validate_postcode(postcode, self.valid_postcodes)

    def detect_intent(self, message: str) -> str:
        """
        Detect the user's intent from their message.

        Args:
            message: User's message

        Returns:
            Detected intent (BUY_HOME, SELL_HOME, GENERAL_QUERY, or INVALID)
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": INTENT_RECOGNITION_TEMPLATE},
                    {"role": "user", "content": message},
                ],
                temperature=0.3,
                max_tokens=50,
                top_p=1,
            )

            intent = completion.choices[0].message.content.strip().upper()
            valid_intents = ["BUY_HOME", "SELL_HOME", "GENERAL_QUERY", "INVALID"]

            # Ensure the intent is valid
            if intent not in valid_intents:
                return "GENERAL_QUERY"

            return intent

        except Exception as e:
            print(f"Error detecting intent: {e}")
            return "GENERAL_QUERY"

    def generate_response(
        self, message: str, intent: str, conversation_history: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a response based on the user's message and intent.

        Args:
            message: User's message
            intent: Detected intent
            conversation_history: List of previous messages

        Returns:
            Generated response
        """
        try:
            # Format conversation history for context (only last 5 messages)
            history_context = "\n".join(
                [
                    f"User: {entry['user_message']}\nAssistant: {entry['assistant_response']}"
                    for entry in conversation_history[-5:]
                ]
            )  # Last 5 messages for context

            # Create the prompt with context
            prompt = RESPONSE_TEMPLATE.format(
                intent=intent, message=message, conversation_history=history_context
            )

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message},
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again."

    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        
        Args:
            message: User's message
            
        Returns:
            Dictionary containing the response and metadata
        """
        next_slot = _get_next_slot(self.slot_state)
        
        if next_slot is None:
            response = "✅ Thank you! Here's the info I have:\n" + "\n".join(
                f"{k}: {v}" for k, v in self.slot_state.items()
            )
        else:
            state_str = "\n".join(f"{k}: {v}" for k, v in self.slot_state.items())
            prompt = SLOT_PROMPT_TEMPLATE.format(
                state=state_str,
                next_field=next_slot,
                user_message=message,
                json=json.dumps(self.slot_state),
            )
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            
            response_raw = completion.choices[0].message.content.strip()
            response_text, extracted_slots = extract_slot_block(response_raw)
            
            # Handle postcode validation
            if "postcode" in extracted_slots and extracted_slots["postcode"]:
                postcode = extracted_slots["postcode"]
                is_valid, formatted_postcode, area = self.validate_user_postcode(postcode)
                
                if not is_valid:
                    if not validate_uk_postcode_format(formatted_postcode):
                        response_text = f"Sorry, '{formatted_postcode}' is not a valid UK postcode format. Please provide a valid UK postcode (e.g., SW1A 1AA)."
                    elif area != "SW":
                        response_text = f"Sorry, we don't cater to the {area} area. We currently only serve the SW (South West London) area. Please call the office on 1800 111 222 to get help."
                    else:
                        response_text = f"Sorry, we don't cater to the postcode {formatted_postcode}. Please call the office on 1800 111 222 to get help."
                    self.slot_state["postcode"] = None  # Reset postcode as it's invalid
                else:
                    self.slot_state["postcode"] = formatted_postcode
                    response_text = f"Great! {formatted_postcode} is in our service area. "
                    if self.slot_state["intent"] == "BUY_HOME":
                        response_text += "You can expect someone to get in touch with you within 24 hours via phone or email. Do you need help with anything else?"
                    else:
                        response_text += "Our team will contact you shortly to discuss your property. Do you need help with anything else?"
            
            # Update other slots
            for key, value in extracted_slots.items():
                if key != "postcode" and key in self.slot_state and self.slot_state[key] is None:
                    self.slot_state[key] = value
            
            response = response_text

        # Store the conversation
        conversation_entry = {
            "timestamp": datetime.now(),
            "user_message": message,
            "assistant_response": response,
            "slots": self.slot_state.copy()
        }
        self.conversation_history.append(conversation_entry)
        
        # Store in vector store if available
        if self.vector_store:
            self.vector_store.add_documents(
                documents=[message, response],
                metadatas=[
                    {"type": "user", "slot_state": self.slot_state},
                    {"type": "assistant", "slot_state": self.slot_state}
                ]
            )
        
        return conversation_entry


def _get_next_slot(state: Dict[str, Any]) -> Optional[str]:
    slot_order = list(state.keys())

    for slot in slot_order:
        if state[slot] is not None:
            continue

        if slot == "property_type":
            print("")
            if state.get("intent") == "BUY_HOME":
                return "property_type"
            else:
                continue  # Skip property_type for SELL_HOME
        return slot  # Next unfilled slot

    return "chat"  # All slots filled


def extract_slot_block(text: str) -> (str, Dict[str, Any]):
    """
    Extracts assistant message and slot values between SLOT_VALUES_START and SLOT_VALUES_END.
    """
    start_tag = "SLOT_VALUES_START"
    end_tag = "SLOT_VALUES_END"

    start = text.find(start_tag)
    end = text.find(end_tag)

    if start == -1 or end == -1 or end <= start:
        return text.strip(), {}

    json_block = text[start + len(start_tag) : end].strip()
    response_text = (text[:start] + text[end + len(end_tag) :]).strip()

    try:
        slots = json.loads(json_block)
        return response_text, slots
    except json.JSONDecodeError:
        return response_text, {}
