from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# 1. Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# 2. Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3. Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 4. Vector Store (Qdrant)
qdrant = Qdrant.from_documents(
    docs,
    embedding_model,
    url="http://localhost:6333",
    collection_name="pdf_collection"
)

# 5. Retriever
retriever = qdrant.as_retriever()

# 6. LLM (Ollama)
llm = Ollama(model="llama3")

# 7. Retrieval + Injection (RAG)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"  # inject retrieved chunks into prompt
)

# 8. Ask question
query = "Summarize the document"
response = qa.run(query)

print("\nAnswer:\n", response)
