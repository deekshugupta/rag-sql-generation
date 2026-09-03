# RAG-SQL-GENERATION

## Components
- main.py - Reading PDF and Performed semantic chunking
- embed_and_store.py - Embedding and Storing in Quadrant DB
- user_query.py - Embedding User Query and Generation SQL

### Code Run

``` 
    uv sync
    source venv/bin/activate
    
    python embed_and_store.py
    python user_query
    

