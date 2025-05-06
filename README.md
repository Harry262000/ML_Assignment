# Real Estate Chatbot

A conversational AI assistant for real estate services, built with Streamlit and OpenAI's GPT-4. The chatbot helps users with buying and selling properties, providing personalized assistance and information.

## Features

- 🤖 Intelligent conversation handling with slot-based approach
- 🏠 Property buying and selling assistance
- 📍 UK postcode validation and area verification
- 💬 Natural language understanding
- 📊 Conversation history tracking
- 🔍 Vector-based memory for context retention

## Project Structure

```
real-estate-chatbot/
├── app/                        # Streamlit application
│   ├── app.py                 # Main application file
│   └── pages/                 # Additional Streamlit pages
│       └── history.py         # Conversation history page
├── src/                       # Source code
│   ├── chatbot.py            # Core chatbot logic
│   ├── memory/               # Memory management
│   │   └── vector_store.py   # Vector store implementation
│   ├── prompts/              # Prompt templates
│   │   └── templates.py      # System prompts and templates
│   └── utils/                # Utility functions
│       └── postcode_validator.py  # UK postcode validation
├── data/                     # Data files
│   └── uk_postcodes.csv      # UK postcodes database
├── .streamlit/               # Streamlit configuration
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd real-estate-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.streamlit/secrets.toml` file with your OpenAI API key:
```toml
OPENAI_API_KEY = "your-api-key-here"
```

5. Run the application:
```bash
streamlit run app/app.py
```

## Usage

1. Open your browser and navigate to `http://localhost:8501`
2. Start a conversation with the chatbot
3. The chatbot will guide you through:
   - Understanding your intent (buying/selling)
   - Collecting necessary information
   - Validating your postcode
   - Providing relevant assistance

## Features in Detail

### Slot-Based Conversation
- Tracks user information through structured slots
- Maintains conversation context
- Handles multiple intents (BUY_HOME, SELL_HOME, GENERAL_QUERY)

### Postcode Validation
- Validates UK postcode format
- Verifies postcode existence
- Checks service area coverage (currently SW London)

### Conversation History
- Tracks all conversations
- Displays slot values and responses
- Allows downloading conversation history

### Vector Store Memory
- Maintains conversation context
- Enables semantic search through past interactions
- Improves response relevance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.