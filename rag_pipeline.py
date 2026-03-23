import os
from dotenv import load_dotenv
from embeddings import search_schema
from db import run_sql

load_dotenv()

# ── LLM CALL ─────────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


# ── PROMPTS ───────────────────────────────────────────────────────────────────

def build_sql_prompt(question: str, schema_context: list[str]) -> str:
    schema_text = "\n".join(schema_context)
    return f"""You are a SQL expert. Given the database schema below, write a valid PostgreSQL SELECT query to answer the user's question.

SCHEMA:
{schema_text}

RULES:
- Only write SELECT queries
- Always add LIMIT 100
- Return ONLY the raw SQL query, no explanation, no markdown, no backticks

USER QUESTION: {question}

SQL QUERY:"""


def build_answer_prompt(question: str, sql: str, results: list[dict]) -> str:
    results_text = str(results[:20])
    return f"""You are a helpful data analyst. A user asked a question, we ran a SQL query and got results.
Answer the user's question in plain English using the data below.
Be concise and clear.

USER QUESTION: {question}

SQL QUERY USED:
{sql}

QUERY RESULTS:
{results_text}

ANSWER:"""


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def ask(question: str) -> dict:
    """
    Full RAG pipeline:
    1. Embed question → find relevant schema
    2. Generate SQL via LLM
    3. Execute SQL on Postgres
    4. Generate natural language answer via LLM
    """
    print(f"\n🔍 Question: {question}")

    # Step 1 — retrieve relevant schema
    schema_context = search_schema(question, top_k=4)
    print(f"📋 Relevant tables found: {len(schema_context)}")
    print("📋 Schema context:")
    for s in schema_context:
        print(f"   {s}")

    # Step 2 — generate SQL
    sql_prompt = build_sql_prompt(question, schema_context)
    sql_query = call_llm(sql_prompt)
    print(f"\n🛠  Generated SQL:\n{sql_query}")

    # Step 3 — execute SQL safely
    try:
        results = run_sql(sql_query)
        print(f"\n✅ Query returned {len(results)} rows")
    except Exception as e:
        print(f"\n❌ SQL execution failed: {e}")
        return {
            "question": question,
            "sql": sql_query,
            "results": [],
            "answer": f"SQL execution failed: {str(e)}"
        }

    # Step 4 — generate natural language answer
    answer_prompt = build_answer_prompt(question, sql_query, results)
    answer = call_llm(answer_prompt)

    print(f"\n💬 Answer: {answer}")

    return {
        "question": question,
        "sql": sql_query,
        "results": results,
        "answer": answer
    }


# ── TEST ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "How many records are in each table?",
        "Who are the top 5 customers by total spending?",
        "What is the best selling product category?"
    ]

    for question in test_questions:
        print("\n" + "="*60)
        response = ask(question)
        print("="*60)