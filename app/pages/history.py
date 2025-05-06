"""
History page for the Real Estate Decision Assistant.
"""
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Conversation History",
    page_icon="📜",
    layout="centered"
)

st.title("📜 Conversation History")

# Initialize session state for history if not exists
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Display conversation history in a table
if st.session_state.conversation_history:
    # Convert history to DataFrame
    df = pd.DataFrame(st.session_state.conversation_history)
    
    # Format timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Display as table
    st.dataframe(
        df,
        column_config={
            "timestamp": "Time",
            "user_message": "User Message",
            "assistant_response": "Assistant Response",
            "intent": "Detected Intent"
        },
        hide_index=True,
    )
    
    # Add download button
    csv = df.to_csv(index=False)
    st.download_button(
        "Download History",
        csv,
        "conversation_history.csv",
        "text/csv",
        key='download-csv'
    )
else:
    st.info("No conversation history available yet.") 