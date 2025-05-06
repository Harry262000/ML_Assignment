"""
Postcode validation utilities for UK postcodes.
"""
import pandas as pd
import re
from typing import List, Optional, Tuple
import os

def load_uk_postcodes(file_path: str = None) -> pd.DataFrame:
    """
    Load UK postcodes from CSV file.
    If no file path is provided, looks for uk_postcodes.csv in the data directory.
    
    Args:
        file_path: Optional path to the CSV file
        
    Returns:
        DataFrame containing UK postcodes
    """
    if file_path is None:
        # Look for uk_postcodes.csv in the data directory
        current_dir = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.join(current_dir, 'data', 'uk_postcodes.csv')
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading UK postcodes: {e}")
        return pd.DataFrame()

def format_postcode(postcode: str) -> str:
    """
    Format postcode to standard UK format (e.g., 'SW1A 1AA').
    
    Args:
        postcode: Raw postcode string
        
    Returns:
        Formatted postcode string
    """
    # Remove all spaces and convert to uppercase
    postcode = postcode.replace(" ", "").upper()
    
    # Insert space before the last 3 characters
    if len(postcode) > 3:
        postcode = postcode[:-3] + " " + postcode[-3:]
    
    return postcode

def validate_uk_postcode_format(postcode: str) -> bool:
    """
    Validate UK postcode format using regex.
    
    Args:
        postcode: Postcode to validate
        
    Returns:
        True if postcode format is valid, False otherwise
    """
    # UK Postcode regex pattern
    pattern = r'^[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}$'
    return bool(re.match(pattern, postcode.upper()))

def validate_postcode(postcode: str, valid_postcodes: List[str] = None) -> Tuple[bool, str, str]:
    """
    Validate if a postcode exists in UK and is in our service area.
    
    Args:
        postcode: Postcode to validate
        valid_postcodes: Optional list of valid postcodes for our service area
        
    Returns:
        Tuple of (is_valid, formatted_postcode, area)
    """
    # First format the postcode
    formatted_postcode = format_postcode(postcode)
    
    # Check if the format is valid
    if not validate_uk_postcode_format(formatted_postcode):
        return False, formatted_postcode, ""
    
    # Load UK postcodes if not provided
    if valid_postcodes is None:
        uk_postcodes_df = load_uk_postcodes()
        if uk_postcodes_df.empty:
            return False, formatted_postcode, ""
        
        # Check if postcode exists in UK
        is_valid_uk = formatted_postcode in uk_postcodes_df['postcode'].str.upper().values
        if not is_valid_uk:
            return False, formatted_postcode, ""
    
    # Get the area code
    area = get_postcode_area(formatted_postcode)
    
    # If we have a list of valid postcodes, check if it's in our service area
    if valid_postcodes:
        is_valid = formatted_postcode in [p.upper() for p in valid_postcodes]
        return is_valid, formatted_postcode, area
    
    return True, formatted_postcode, area

def get_postcode_area(postcode: str) -> str:
    """
    Extract the area code from a UK postcode (e.g., 'SW' from 'SW1A 1AA').
    
    Args:
        postcode: UK postcode
        
    Returns:
        Area code
    """
    # Remove spaces and get the first part before any numbers
    area = re.match(r'^[A-Z]+', postcode.replace(" ", "").upper())
    return area.group(0) if area else "" 