import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import uuid

load_dotenv()

index_name = "doom-ai-index"
vector_store = None

groq_api_key = os.environ['GROQ_API_KEY']
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
pinecone_api_key = os.getenv("PINECONE_API_KEY")

def get_text_chunks(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(document)
    return chunks

def get_vectorstore_from_url(url: str):
    global vector_store
    identity = str(uuid.uuid4())
    loader = WebBaseLoader(url)
    document = loader.load()
    doc_chunks = get_text_chunks(document)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=identity)
    vector_store.from_documents(documents=doc_chunks, embedding=embeddings, index_name=index_name, namespace=identity)
    return vector_store
    
def get_context_retriever_chain():
    llm = ChatGroq(
        groq_api_key = groq_api_key,
        model_name = "llama3-8b-8192"
    )
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatGroq(
        groq_api_key = groq_api_key,
        model_name = "llama3-8b-8192"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based solely on the below context:\n\n{context} .If the answer is not inthe provided context, please answer with 'Sorry, I don't know the answer to that question.', don't provide any other wrong answer or any answer self made by you from any other resources.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)
