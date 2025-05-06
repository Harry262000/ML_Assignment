import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


class Chatbot:
    def __init__(self, api_key, postcode_validator):
        # Initialize LLM
        self.llm = ChatGroq(client=client, model_name="x-ai/grok-3-mini-beta", api_key=api_key)

        # Initialize other components
        self.postcode_validator = postcode_validator
        self.memory = ConversationBufferMemory(return_messages=True)
        self.current_user_id = 123  # Start with user ID 123
        self.state = {
            "conversation_state": "start",
            "intent": None,
            "home_type": None,
            "name": None,
            "phone": None,
            "email": None,
            "budget": None,
            "postcode": None
        }

        # Define prompt templates
        self.INTENT_PROMPT = PromptTemplate.from_template(
            "Classify the intent of the following message as 'buy', 'sell', or 'other': {input}"
        )
        self.HOME_TYPE_PROMPT = PromptTemplate.from_template(
            "The user was asked: 'Are you interested in a new home or a resale home?'. Classify their response as 'new home', 'resale', or 'unclear'. User's response: {input}"
        )
        self.USER_DETAILS_PROMPT = PromptTemplate.from_template(
            "The user was asked: 'May I have your name, phone number, and email address, please?'. Extract the name, phone, and email from their response and output as JSON: {{'name': value, 'phone': value, 'email': value, 'is_question': boolean}}. If any detail is missing or the response is a question, set 'is_question' to true and use null for missing fields. User's response: {input}"
        )
        self.ENTITY_PROMPT = PromptTemplate.from_template(
            "Extract the budget and postcode from the following message and output as JSON: {{'budget': value, 'postcode': value}}. If not present, use null. Message: {input}"
        )
        self.FALLBACK_PROMPT = PromptTemplate.from_template(
            "The user was asked: '{question}'. Their response was: '{input}'. This doesn't match the expected format. Generate a polite response to guide the user back to providing the requested information, or answer their question if applicable."
        )

    def extract_intent(self, user_input):
        # Handle numeric input
        user_input = user_input.strip()
        if user_input == "1":
            return "buy"
        elif user_input == "2":
            return "sell"

        # Handle natural language input
        prompt = self.INTENT_PROMPT.format(input=user_input)
        # Directly call the client's chat.completions.create method with the necessary parameters
        return self.llm.client.chat.completions.create(
            model=self.llm.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # You can adjust these parameters as needed
            max_tokens=1024,
            top_p=1,
            stream=False
        ).choices[0].message.content.lower()

    def extract_entities(self, user_input):
        prompt = self.ENTITY_PROMPT.format(input=user_input)
        response = self.llm.invoke(prompt).content
        try:
            entities = json.loads(response)
            budget = float(entities.get("budget")) if entities.get("budget") else None
            postcode = entities.get("postcode")
        except (json.JSONDecodeError, ValueError):
            budget = None
            postcode = None
        return budget, postcode

    def extract_home_type(self, user_input):
        prompt = self.HOME_TYPE_PROMPT.format(input=user_input)
        response = self.llm.invoke(prompt).content.lower()
        return response

    def extract_user_details(self, user_input):
        prompt = self.USER_DETAILS_PROMPT.format(input=user_input)
        response = self.llm.invoke(prompt).content
        try:
            details = json.loads(response)
            name = details.get("name")
            phone = details.get("phone")
            email = details.get("email")
            is_question = details.get("is_question", False)
        except (json.JSONDecodeError, ValueError):
            name, phone, email, is_question = None, None, None, True
        return name, phone, email, is_question

    def handle_fallback(self, user_input, question):
        prompt = self.FALLBACK_PROMPT.format(question=question, input=user_input)
        return self.llm.invoke(prompt).content

    def process_input(self, user_input, user_id):
        # Handle reassistance state
        if self.state["conversation_state"] == "reassistance":
            if user_input.lower() in ["no", "goodbye"]:
                self.state["conversation_state"] = "start"
                self.memory.chat_memory.messages = []  # Reset memory
                self.state["intent"] = None
                self.state["home_type"] = None
                self.state["name"] = None
                self.state["phone"] = None
                self.state["email"] = None
                self.state["budget"] = None
                self.state["postcode"] = None
                self.current_user_id += 1
                return "Thank you for chatting with us. Goodbye. \n\n--- New Chat Started ---\nUser ID: {}\nHello! How can I help you today?\n1. You want to buy a house\n2. You want to sell a house".format(
                    self.current_user_id)
            else:
                # Treat as a new query without resetting user ID
                self.state["conversation_state"] = "start"
                return self.process_input(user_input,
                                          user_id)  # Recursively process new input

        # State machine to handle conversation flow
        if self.state["conversation_state"] == "start":
            intent = self.extract_intent(user_input)
            if "buy" in intent:
                self.state["intent"] = "buy"
                self.state["conversation_state"] = "home_type"
                return "Are you interested in a new home or a resale home?"
            elif "sell" in intent:
                self.state["intent"] = "sell"
                self.state["conversation_state"] = "user_details"
                return "May I have your name, phone number, and email address, please?"
            else:
                return "Sorry, I didn’t understand. Hello! How can I help you today?\n1. You want to buy a house\n2. You want to sell a house"

        elif self.state["conversation_state"] == "home_type":
            home_type = self.extract_home_type(user_input)
            if home_type == "new home":
                self.state["home_type"] = "new home"
            elif home_type == "resale":
                self.state["home_type"] = "resale"
            else:
                return "Are you interested in a new home or a resale home?"
            self.state["conversation_state"] = "user_details"
            return "May I have your name, phone number, and email address, please?"

        elif self.state["conversation_state"] == "user_details":
            name, phone, email, is_question = self.extract_user_details(
                user_input)
            if is_question or not all([name, phone, email]):
                question = "May I have your name, phone number, and email address, please?"
                return self.handle_fallback(user_input, question)
            self.state["name"] = name
            self.state["phone"] = phone
            self.state["email"] = email
            self.state["conversation_state"] = "budget_postcode"
            if self.state["intent"] == "buy":
                return f"Thank you, {name}! What is your budget? What is your postcode?"
            else:  # sell
                return f"Thank you, {name}! What is your postcode?"

        elif self.state["conversation_state"] == "budget_postcode":
            budget, postcode = self.extract_entities(user_input)
            self.state["budget"] = budget
            self.state["postcode"] = postcode

            # If any required info is missing, ask again
            if self.state["intent"] == "buy" and (not budget or not postcode):
                return "What is your budget? What is your postcode?"
            if self.state["intent"] == "sell" and not postcode:
                return "What is your postcode?"

            # Validate inputs
            if self.state["intent"] == "buy":
                if budget < 1000000:
                    self.state["conversation_state"] = "start"
                    self.current_user_id += 1
                    return "Sorry, we don’t cater to budgets below 1 million. Please call the office on 1800 111 222 to get help.\n\n--- New Chat Started ---\nUser ID: {}\nHello! How can I help you today?\n1. You want to buy a house\n2. You want to sell a house".format(
                        self.current_user_id)
            if not self.postcode_validator.is_valid(postcode):
                self.state["conversation_state"] = "start"
                self.current_user_id += 1
                return "Sorry, we don’t cater to Post codes that you provided. Please call the office on 1800 111 222 to get help.\n\n--- New Chat Started ---\nUser ID: {}\nHello! How can I help you today?\n1. You want to buy a house\n2. You want to sell a house".format(
                        self.current_user_id)


            # Move to reassistance state
            self.state["conversation_state"] = "reassistance"
            self.memory.save_context({"input": user_input},
                                     {"output": "Callback assured"})
            return "I can expect someone to get in touch with you within 24 hours via phone or email. Is there anything I can help you with?"


# Example usage (requires api_key and postcode_validator)
# from your_postcode_validator import PostcodeValidator
chatbot = Chatbot(api_key=api_key, postcode_validator=PostcodeValidator())