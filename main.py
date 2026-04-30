from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# -----------------------
# Load + Process PDF
# -----------------------
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

qdrant = Qdrant.from_documents(
    docs,
    embedding_model,
    url="http://localhost:6333",
    collection_name="pdf_collection"
)

retriever = qdrant.as_retriever()

llm = Ollama(model="llama3")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# -----------------------
# FastAPI App
# -----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.post("/ask")
def ask(q: Query):
    answer = qa.run(q.query)
    return {"answer": answer}