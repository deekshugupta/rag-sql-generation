import uuid
import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv
from main import DynamicSchemaChunker
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    PointStruct, 
    PayloadSchemaType
)

load_dotenv()

HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))

def ingest_chunks_to_qdrant(
    chunks: List[Dict[str, Any]], 
    collection_name: str = "travel_rag_schema",
    embedding_model: str = "text-embedding-3-small"
):
    """
    Embeds Markdown schema chunks and ingests them into a Qdrant instance with indexed metadata.
    """
    # 1. Initialize Clients
    qdrant = QdrantClient(host=HOST, port=PORT)
    openai_client = OpenAI()

    # 2. Define payload indexes for fast filtered searches
    filterable_fields = {
        "metadata.database": PayloadSchemaType.KEYWORD,
        "metadata.table_name": PayloadSchemaType.KEYWORD,
        "metadata.columns": PayloadSchemaType.KEYWORD,
        "metadata.synonyms": PayloadSchemaType.KEYWORD,
        "metadata.related_tables": PayloadSchemaType.KEYWORD,
    }

    # 3. Create Collection and Payload Indexes if missing
    if not qdrant.collection_exists(collection_name=collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"Created collection: '{collection_name}'")

        # Register metadata schema indexes
        for field_name, field_schema in filterable_fields.items():
            qdrant.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema
            )
        print("Successfully created metadata payload indexes.")

    # 4. Batch Generate Embeddings & Prepare Point Structures
    points = []
    print(f"Generating embeddings for {len(chunks)} chunks...")

    for chunk in chunks:
        text = chunk.get("text", "")
        if not text.strip():
            continue  # Skip empty text chunks

        raw_meta = chunk.get("metadata", {})
        
        payload = {
            "chunk_id": chunk.get("id"),
            "text": text,
            "metadata": {
                "database": str(raw_meta.get("database", "default")),
                "table_name": str(raw_meta.get("table_name", "")),
                "table_alias": str(raw_meta.get("alias", "")),
                "primary_keys": list(raw_meta.get("primary_keys", [])),
                "columns": [str(col).lower() for col in raw_meta.get("columns", [])],
                "synonyms": [str(syn).lower() for syn in raw_meta.get("synonyms", [])],
                "related_tables": list(raw_meta.get("related_tables", [])),
                "has_foreign_keys": bool(raw_meta.get("has_foreign_keys", False))
            }
        }

        # Generate embedding
        response = openai_client.embeddings.create(
            input=text,
            model=embedding_model
        )
        vector = response.data[0].embedding

        # Deterministic UUID generation
        chunk_id = str(chunk.get("id") or uuid.uuid4())
        point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        )

    # 5. Batch Upsert Vectors
    if points:
        qdrant.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Successfully ingested {len(points)} chunks into '{collection_name}'.")

if __name__ == "__main__":
    chunker = DynamicSchemaChunker()
    auto_chunks = chunker.process_pdf()
    ingest_chunks_to_qdrant(chunks=auto_chunks)