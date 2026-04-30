🔄 How It Works (Pipeline)
1. PDF → Text
Loads your 2-page PDF
2. Chunking
Splits into smaller pieces (important for retrieval)
3. Embedding
Converts chunks → vectors
4. Vector Storage
Stored in Qdrant
5. Retrieval
Finds relevant chunks for query
6. Injection (RAG)
Injects retrieved chunks into prompt
7. LLM Response
Ollama generates answer based on context
