#!/usr/bin/env python3
"""
Utility functions for seeding DynamoDB from CSV data
"""

def generate_user_id(first, last, cc_num):
    """
    Generate a consistent user_id from name and card number.
    This ensures the same person always gets the same user_id across seeding scripts.
    """
    return f"user_{first.lower()}_{last.lower()}_{str(abs(hash(str(cc_num))))[:6]}"




