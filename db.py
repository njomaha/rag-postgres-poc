import sqlalchemy
from config import DB_URL

engine = sqlalchemy.create_engine(DB_URL)

def get_schema_as_text() -> list[dict]:
    """
    Reads all tables and columns from your Postgres DB
    and returns them as embeddable text chunks.
    """
    schema_docs = []

    with engine.connect() as conn:
        # Get all tables
        tables = conn.execute(sqlalchemy.text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)).fetchall()

        for (table_name,) in tables:
            # Get columns for each table
            columns = conn.execute(sqlalchemy.text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = :table
                ORDER BY ordinal_position
            """), {"table": table_name}).fetchall()

            col_text = ", ".join([f"{col} ({dtype})" for col, dtype in columns])

            # This is what gets embedded
            doc_text = f"Table: {table_name}. Columns: {col_text}"

            schema_docs.append({
                "id": table_name,
                "text": doc_text,
                "table": table_name
            })

    return schema_docs


def run_sql(query: str) -> list[dict]:
    """Safely run SELECT queries only."""
    query = query.strip()
    if not query.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(query))
        rows = result.fetchmany(100)    # limit rows for POC
        columns = result.keys()
        return [dict(zip(columns, row)) for row in rows]