"""
Tests for utility functions.
"""
import pytest
from src.utils.postcode_validator import validate_postcode, load_postcodes

def test_validate_postcode():
    """Test postcode validation."""
    valid_postcodes = ["SW1A 1AA", "EC1A 1BB", "W1A 1AA"]
    
    # Test valid postcodes
    assert validate_postcode("SW1A 1AA", valid_postcodes) is True
    assert validate_postcode("EC1A 1BB", valid_postcodes) is True
    
    # Test invalid postcodes
    assert validate_postcode("INVALID", valid_postcodes) is False
    assert validate_postcode("", valid_postcodes) is False

def test_load_postcodes():
    """Test loading postcodes from CSV."""
    # TODO: Add test for loading postcodes from CSV file
    pass 