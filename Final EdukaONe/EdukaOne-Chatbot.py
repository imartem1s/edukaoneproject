import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Embedding object
embeddings = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model='models/embedding-001')

# Pinecone object
pc = Pinecone(api_key=PINECONE_API_KEY)


# Vectorstore object
pinecone_index = 'edukaone-new'
vectorstore = PineconeVectorStore(index_name=pinecone_index, embedding=embeddings)

# Retriver object
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Output parser object
parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

message_template = """
You are One, a clever and friendly assistant who works for EdukaOne.
You will teach every student about science nature be it mathematics, physics, biology, and chemsitry.
Your job is to teach, explain, and answer their question about science.
You also will help student to tackle their problem,
help them to answer their curiousity towards science. 
Once you have the user's answer, you will explain further more so the student will become excited to scientific topics.
Answer based on the context provided. 
You are physics expert. Please answer my question based on context provide. 
If you cannot find the solution in the context and you dont know the answer, answer with your base knowledge about science
and make it fun and creative so the user will understand science better

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_messages([("human", message_template)])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

user_prompt = input("Ask One: ")
response = chain.invoke(user_prompt)
print(response)