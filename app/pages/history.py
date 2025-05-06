"""
History page for the Real Estate Decision Assistant.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import json

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
    
    # Extract slot values into separate columns
    df['slots'] = df['slots'].apply(lambda x: json.dumps(x, indent=2) if isinstance(x, dict) else x)
    
    # Display as table with better formatting
    st.dataframe(
        df,
        column_config={
            "timestamp": st.column_config.TextColumn(
                "Time",
                width="medium"
            ),
            "user_message": st.column_config.TextColumn(
                "User Message",
                width="large"
            ),
            "assistant_response": st.column_config.TextColumn(
                "Assistant Response",
                width="large"
            ),
            "slots": st.column_config.TextColumn(
                "Slot Values",
                width="large",
                help="Current state of all slots during this conversation turn"
            )
        },
        hide_index=True,
        height=400
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
    
    # Add a section to view slot values in a more readable format
    st.subheader("📊 Slot Values Summary")
    if not df.empty:
        # Get the latest slot values
        latest_slots = df['slots'].iloc[-1]
        if isinstance(latest_slots, str):
            try:
                latest_slots = json.loads(latest_slots)
            except json.JSONDecodeError:
                latest_slots = {}
        
        # Display slot values in a more readable format
        cols = st.columns(3)
        for i, (key, value) in enumerate(latest_slots.items()):
            with cols[i % 3]:
                st.metric(
                    label=key.replace("_", " ").title(),
                    value=str(value) if value is not None else "Not set"
                )
else:
    st.info("No conversation history available yet.") 