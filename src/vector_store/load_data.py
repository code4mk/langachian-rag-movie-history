from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# Set up API keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")


def create_vectorstore():
    # Load and split the documents
    loader = TextLoader("movie_history.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Set up API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")

    # Set up OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # Create or connect to the index
    index_name = "movie-quiz"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=1536, metric="cosine")

    # Create the vector store
    vectorstore = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    print("Vector store created successfully!")

if __name__ == "__main__":
    create_vectorstore()