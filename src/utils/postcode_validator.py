"""
Postcode validation utilities.
"""
import pandas as pd
from typing import List, Optional

def load_postcodes(file_path: str) -> List[str]:
    """
    Load postcodes from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing postcodes
        
    Returns:
        List of valid postcodes
    """
    try:
        df = pd.read_csv(file_path)
        return df['postcode'].tolist()
    except Exception as e:
        print(f"Error loading postcodes: {e}")
        return []

def validate_postcode(postcode: str, valid_postcodes: List[str]) -> bool:
    """
    Validate if a postcode is in the list of valid postcodes.
    
    Args:
        postcode: Postcode to validate
        valid_postcodes: List of valid postcodes
        
    Returns:
        True if postcode is valid, False otherwise
    """
    return postcode.upper() in [p.upper() for p in valid_postcodes] 