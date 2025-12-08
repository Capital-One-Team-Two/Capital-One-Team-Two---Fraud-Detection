ACCOUNTS = [
    {
        'user_id': 'user_001',
        'name': 'John Doe',
        'phone_number': '+16082177160',  # Verified phone number
        'created_at': '2024-01-01T00:00:00Z',
        'account_status': 'active'
    },
    {
        'user_id': 'user_002',
        'name': 'Jane Smith',
        'phone_number': '+16082177160',  # Verified phone number
        'created_at': '2024-01-15T00:00:00Z',
        'account_status': 'active'
    },
    {
        'user_id': 'user_003',
        'name': 'Bob Johnson',
        'phone_number': '+16082177160',  # Verified phone number
        'created_at': '2024-02-01T00:00:00Z',
        'account_status': 'active'
    }
]

SAMPLE_TRANSACTIONS = [
    {
        'transaction_id': 'txn_001',
        'user_id': 'user_001',
        'amount': 45.99,
        'merchant': 'Amazon',
        'location': 'Seattle, WA',
        'timestamp': '2024-10-14T10:30:00Z',
        'card_number': '****1234'
    },
    {
        'transaction_id': 'txn_002',
        'user_id': 'user_001',
        'amount': 1500.00,
        'merchant': 'Unknown Vendor',
        'location': 'Lagos, Nigeria',
        'timestamp': '2024-10-14T11:00:00Z',
        'card_number': '****1234'
    },
    {
        'transaction_id': 'txn_003',
        'user_id': 'user_002',
        'amount': 25.50,
        'merchant': 'Starbucks',
        'location': 'Madison, WI',
        'timestamp': '2024-10-14T08:15:00Z',
        'card_number': '****5678'
    }
]