import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain_pinecone import PineconeVectorStore
import uuid

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
pinecone_api_key = os.getenv("PINECONE_API_KEY")

index_name = "doom-ai-index"
identity = None

def get_pdf_text(pdf_doc):
    pdf_text = ""
    pdf = PdfReader(pdf_doc)
    for page in pdf.pages:
        pdf_text += page.extract_text()
    return pdf_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    global identity
    identity = str(uuid.uuid4())
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # vector_store.save_local("faiss_index")
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    vector_store.add_texts(text_chunks, namespace=identity)
    
def get_conversational_chain():   
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    the provided context, please answer with "Sorry, I don't know the answer to that question.", don't provide any other wrong answer.\n\n
    Context:\n {context}?\n
    Question:\n {question}\n
    
    Answer:
    """
    llm = ChatGroq(
            groq_api_key = groq_api_key,
            model_name = "llama3-8b-8192"
    )
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # docs = new_db.similarity_search(user_question)
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    docs = vector_store.similarity_search(user_question, k=5, namespace=identity)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']