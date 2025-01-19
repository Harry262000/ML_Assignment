import streamlit as st
import torch
from model import CpGPredictor
from biopython.Seq import Seq

# Load your trained model
model = CpGPredictor(input_size, hidden_size, num_layers, dropout)
model.load_state_dict(torch.load('path/to/your/trained/model.pth'))
model.eval()

st.title('CpG Island Predictor')

# Get DNA sequence input from the user
sequence = st.text_area('Enter DNA sequence:')

# Preprocess the sequence
def preprocess_sequence(sequence):
    # Convert the DNA sequence to an integer sequence using Biopython
    dna_seq = Seq(sequence)
    int_seq = list(dna_seq.translate(to=dna2int))

    # Pad the sequence to a fixed length (e.g., 128)
    padded_seq = torch.zeros(1, 128)
    padded_seq[:len(int_seq)] = torch.tensor(int_seq)

    return padded_seq

if st.button('Predict'):
    if sequence:
        # Preprocess the input sequence
        preprocessed_sequence = preprocess_sequence(sequence)

        # Make prediction
        with torch.no_grad():
            prediction = model(preprocessed_sequence)

        # Display the prediction
        st.write(f'Predicted CpG count: {prediction.item():.2f}')
    else:
        st.write('Please enter a DNA sequence.')