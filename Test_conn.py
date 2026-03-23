from dotenv import load_dotenv
import os

load_dotenv()

# Debug — print what's being read from .env
print("DEBUG — ENV VALUES:")
print(f"  DB_HOST     = {os.getenv('DB_HOST')}")
print(f"  DB_PORT     = {os.getenv('DB_PORT')}")
print(f"  DB_NAME     = {os.getenv('DB_NAME')}")
print(f"  DB_USER     = {os.getenv('DB_USER')}")
print(f"  DB_PASSWORD = {'SET' if os.getenv('DB_PASSWORD') else 'MISSING'}")
print()