import os
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI
from sentence_transformers import CrossEncoder

load_dotenv()


HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))

# Initialize clients globally to reuse connection pools
qdrant_client = QdrantClient(host=HOST, port=PORT)
openai_client = OpenAI()

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_relevant_schema(
    user_query: str, 
    collection_name: str = "travel_rag_schema", 
    top_k: int = 20,
    limit: int = 5
) -> str:
    """
    Retrieves top matching schema contexts from Qdrant based on query embedding,
    fetching top_k candidates and reranking them down to `limit` results.
    """
    # 1. Embed user prompt
    query_vector = openai_client.embeddings.create(
        input=user_query,
        model="text-embedding-3-small"
    ).data[0].embedding
    # 2. Similarity search using updated Qdrant API - fetch top_k candidates
    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )
    # 3. Rerank candidates using cross-encoder for improved relevance
    candidates = response.points
    if candidates:
        pairs = [(user_query, (point.payload or {}).get("text", "")) for point in candidates]
        rerank_scores = reranker_model.predict(pairs)
        scored_candidates = list(zip(candidates, rerank_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = [point for point, score in scored_candidates[:limit]]

    # 4. Build grounding context for the LLM
    retrieved_context = []
    for point in candidates:
        payload = point.payload or {}
        meta = payload.get("metadata", {})
        
        table_name = meta.get("table_name", "Unknown")
        db_name = meta.get("database", "default")
        columns = ", ".join(meta.get("columns", []))
        text_chunk = payload.get("text", "")
        table_info = (
            f"Table: {table_name} (Database: {db_name})\n"
            f"Columns: {columns}\n"
            f"Schema Details:\n{text_chunk}\n"
        )
        retrieved_context.append(table_info)
    return "\n---\n".join(retrieved_context)

def generate_sql_query(user_query: str, schema_context: str) -> str:
    """
    Generates a syntactically correct SQL query grounded in the retrieved schema context.
    """
    system_prompt = f"""
You are an expert SQL Generator. Your task is to generate a syntactically correct SQL query based ONLY on the schema context provided below.
Rules:
1. Use ONLY the tables and columns mentioned in the Context.
2. Output ONLY the raw SQL query. Do not add conversational explanations or Markdown wrappers.
3. Handle JOINs correctly based on primary and foreign keys provided in the context.
Retrieved Schema Context:
{schema_context}
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.0
    )
    sql_output = response.choices[0].message.content.strip()
    
    # cleaned_sql = re.sub(r"^```(?:sql)?\n|\n```$", "", sql_output, flags=re.IGNORECASE).strip()
    return sql_output

def text_to_sql_pipeline(user_query: str) -> str:
    """
    End-to-End Orchestrator: Query -> Vector Search -> Context -> SQL
    """
    print(f"\n[1/2] Searching Qdrant schema for: '{user_query}'...")
    schema_context = retrieve_relevant_schema(user_query)
    
    if not schema_context:
        return "-- Error: No relevant schema found in vector store."
    print("[2/2] Generating SQL query with GPT-4o-mini...")
    generated_sql = generate_sql_query(user_query, schema_context)
    
    return generated_sql

if __name__ == "__main__":
    # Example natural language request
    sample_query = "give hotel names book for john"
    
    sql = text_to_sql_pipeline(sample_query)
    
    print("\n=== Final Generated SQL ===")
    print(sql)