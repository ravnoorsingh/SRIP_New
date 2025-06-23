# Loader
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

# Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Gemini Embeddings and Model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Qdrant vector storage
from langchain_qdrant import QdrantVectorStore

from dotenv import load_dotenv
load_dotenv()

# Load and process PDF
pdf_path = Path(__file__).parent/"nodeJS.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(documents=docs)

# Initialize Gemini Embeddings
embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

#Initialize vector store (uncomment to inject documents)
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="new",
    embedding=embedder
)
print("Injection Done")

# Initialize retriever
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="new",
    embedding=embedder
)

# Query processing
user_query = input('> ')
relevant_chunks = retriever.similarity_search(query=user_query)

# Format context
context_str = "\n\n".join([doc.page_content for doc in relevant_chunks])

# System prompt with retrieved context
SYSTEM_PROMPT = f"""
You are a helpful AI Assistant who responds based on the available context.

Context:
{context_str}
"""

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

# Generate response
response = model.invoke([
    ("system", SYSTEM_PROMPT),
    ("user", user_query)
])

print("Gemini Output:\n", response.content)
