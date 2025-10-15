import boto3
import json
from sample_data import ACCOUNTS, SAMPLE_TRANSACTIONS
import sys
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('AccountsTable')

def seed_accounts_table(table_name="FraudDetection-Accounts"):

    table = dynamodb.Table(table_name)

    print(f"Seeding {table_name}...")

    for account in ACCOUNTS:
        table.put_item(Item=account)

    print("Accounts table seeded successfully!")

def seed_transactions_table(table_name='FraudDetection-Transactions'):
    table = dynamodb.Table(table_name)
    
    print(f"Seeding {table_name}...")
    for transaction in SAMPLE_TRANSACTIONS:
        item = json.loads(json.dumps(transaction), parse_float=Decimal)
        table.put_item(
            Item=item
        )
    
    print("Transactions table seeded successfully!")


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python seed_database.py [accounts|transactions|all]")
        sys.exit(1)
    
    option = sys.argv[1].lower()
    
    if option == 'accounts' or option == 'all':
        seed_accounts_table()
    
    if option == 'transactions' or option == 'all':
        seed_transactions_table()
    
    print("\nDatabase seeding completed!")
