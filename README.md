# Real Estate Decision Assistant

A conversational AI assistant that helps users navigate through the home buying and selling process. The system uses advanced language models and vector databases to provide context-aware responses and validate user inputs.

## Features

- Intent understanding (buy/sell)
- New home and re-sale logic
- Input validation
- Postcode verification
- Context-aware responses
- Vector store for conversation memory

## Project Structure

```
real-estate-chatbot/
│
├── src/                        # Source code for the chatbot logic
│   ├── __init__.py            # Makes src a Python package
│   ├── chatbot.py             # Core chatbot logic
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── postcode_validator.py
│   │   └── data_loader.py
│   ├── memory/               # Conversation memory
│   │   ├── __init__.py
│   │   └── vector_store.py
│   └── prompts/             # Prompt templates
│       ├── __init__.py
│       └── templates.py
│
├── app/                     # Streamlit app
│   ├── app.py              # Main app entry point
│   └── pages/              # Additional pages
│
├── data/                   # Data storage
│   ├── postcodes.csv      # Valid postcodes
│   └── chroma_db/         # Vector store
│
├── tests/                 # Unit tests
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd real-estate-chatbot
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_api_key_here
```

## Running the Application

To run the Streamlit app:
```bash
streamlit run app/app.py
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.