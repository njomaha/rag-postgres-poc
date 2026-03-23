import sys

print("Running setup checks...\n")

# ── 1. ChromaDB ──────────────────────────────────────────────────────────────
try:
    import chromadb
    client = chromadb.Client()
    col = client.create_collection("test")
    print("✅ ChromaDB OK")
except Exception as e:
    print(f"❌ ChromaDB FAILED: {e}")

# ── 2. Sentence Transformers (Embeddings) ────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode("hello world")
    print(f"✅ Embeddings OK — vector size: {len(vec)}")
except Exception as e:
    print(f"❌ Embeddings FAILED: {e}")

# ── 3. PostgreSQL Connection ─────────────────────────────────────────────────
try:
    import sqlalchemy
    from dotenv import load_dotenv
    import os

    load_dotenv()

    db_url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    engine = sqlalchemy.create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("SELECT current_database()"))
        db_name = result.fetchone()[0]
        print(f"✅ PostgreSQL OK — connected to: {db_name}")
except Exception as e:
    print(f"❌ PostgreSQL FAILED: {e}")

# ── 4. Check Tables Exist ────────────────────────────────────────────────────
try:
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """))
        tables = [row[0] for row in result.fetchall()]
        print(f"✅ Tables found: {tables}")
except Exception as e:
    print(f"❌ Table check FAILED: {e}")

# ── 5. Package Check ─────────────────────────────────────────────────────────
packages = ["streamlit", "langchain", "dotenv"]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg} OK")
    except ImportError:
        print(f"❌ {pkg} NOT installed — run: pip install {pkg}")

print("\nSetup check complete!")