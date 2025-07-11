# Loader
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

#Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
# Embeddings
from langchain_openai import OpenAIEmbeddings

# qdrant vector storage
from langchain_qdrant import QdrantVectorStore

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

pdf_path = Path(__file__).parent/"nodeJS.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200 # every chuck will be overlapping by 200 chucks with it's adjecent splitter
)

split_docs = text_splitter.split_documents(documents=docs)

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
)

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    url="http://localhost:6333",
    collection_name="new1",
    embedding=embedder
)

vector_store.add_documents(documents=split_docs)
print("Injection Done")

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="srip",
    embedding=embedder
)
user_query = input('> ')
# getting Relevent Chunks
relevent_chunks = retriver.similarity_search(
    query=user_query
)

# print("Relevent Chunks ",relevent_chunks)

SYSTEM_PROMPTS = f"""
You are an helpful AI Assistant who responds base of the available context.

Context:
{relevent_chunks}
"""


client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPTS},
        {"role": "user", "content": user_query}
    ]
)

print("GPT Output:\n", response.choices[0].message.content)




