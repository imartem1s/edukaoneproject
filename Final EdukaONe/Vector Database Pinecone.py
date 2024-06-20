import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone,ServerlessSpec


#Loading the environment
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def read_doc(directory):
    file_loader = PyPDFLoader(directory)
    documents=file_loader.load()
    return documents

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

if 'edukaone-new' not in pc.list_indexes().names():
    pc.create_index(
        name='edukaone-new',
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

for file in os.listdir("data/"):
    doc = read_doc(f'data/{file}')

    documents = chunk_data(docs=doc)

    embeddings = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model='models/embedding-001')


    vectorstore_from_docs = PineconeVectorStore.from_documents(
            documents=documents,
            index_name='edukaone-new',
            embedding=embeddings
        )
    print(file, "done")