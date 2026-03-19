from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import CrossEncoder

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment")

# FastAPI app
app = FastAPI(title="Render RAG API")
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Request schema
class QueryRequest(BaseModel):
    query: str


print("Loading models...")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# Cross Encoder reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Groq client
client = Groq(api_key=GROQ_API_KEY)

# Paths
FAISS_PATH = "faiss_index"
DOCS_PATH = "documents/usa2.txt"

# Create or load FAISS index
if os.path.exists(FAISS_PATH):
    print("Loading existing FAISS index...")
    vector_store = FAISS.load_local(
        FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

else:
    print("Creating new FAISS index...")

    loader = TextLoader(DOCS_PATH, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(
        split_docs,
        embedding_model
    )

    vector_store.save_local(FAISS_PATH)

    print("FAISS index created and saved.")

# Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

print("RAG system ready!")


# ---------------------------
# Reranking Function
# ---------------------------

def rerank(query, docs, top_k=3):

    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in ranked[:top_k]]


# ---------------------------
# Groq LLM Generation
# ---------------------------

def generate_answer(query, context):

    if not context.strip():
        return "Not found in the document."

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content": "Answer strictly from the provided context."
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{query}

Answer in 1-2 lines. If not found say 'Not found in document'.
"""
            }
        ]

    )

    return response.choices[0].message.content


# ---------------------------
# API Endpoint
# ---------------------------

@app.post("/ask")

def ask_question(request: QueryRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    # Retrieve documents
    docs = retriever.get_relevant_documents(query)

    # Rerank
    reranked_docs = rerank(query, docs)

    # Build context
    context = "\n\n".join(
        [doc.page_content for doc in reranked_docs]
    )

    # Generate answer
    answer = generate_answer(query, context)

    return {
        "query": query,
        "answer": answer,
        "sources": [
            doc.metadata.get("source", "unknown")
            for doc in reranked_docs
        ]
    }
