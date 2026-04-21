# ...existing code...
import os
import uuid
import traceback
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# configuration via env, sensible defaults
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", None)

GEN_MODEL = os.getenv("GOOGLE_GENERATIVE_MODEL","models/gemini-2.5-flash")
# embedding model: use Gemini embedding resource name by default
EMBED_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
# keep the Groq LLM config as before (change via env if needed)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

index_name = os.getenv("PINECONE_INDEX", "doom-ai-index")
identity = None

# configure Google client
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# optional: initialize Pinecone (no-op if key missing)
try:
    if PINECONE_API_KEY:
        import pinecone
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
except Exception as e:
    print("Warning: pinecone.init failed:", e, flush=True)

print("groq_pdf_chat starting", flush=True)
print("GROQ_API_KEY present:", bool(GROQ_API_KEY), flush=True)
print("GOOGLE_API_KEY present:", bool(GOOGLE_API_KEY), flush=True)
print("Using embedding model:", EMBED_MODEL, flush=True)
print("Using Groq model:", GROQ_MODEL, flush=True)

def get_pdf_text(pdf_doc_path: str) -> str:
    pdf_text = ""
    pdf = PdfReader(pdf_doc_path)
    print(f"Opened PDF {pdf_doc_path}, pages={len(pdf.pages)}", flush=True)
    for i, page in enumerate(pdf.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"Warning: extract_text failed on page {i}: {e}", flush=True)
            text = ""
        pdf_text += text + "\n"
    print("Extracted text length:", len(pdf_text), flush=True)
    return pdf_text

def get_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    print("Created chunks:", len(chunks), flush=True)
    return chunks

def get_vector_store(text_chunks: list) -> None:
    """
    Create and save a local FAISS index using the configured embedding model.
    """
    global identity
    identity = str(uuid.uuid4())
    try:
        print("Creating embeddings with model:", EMBED_MODEL, flush=True)
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        print("Saved FAISS index to 'faiss_index'", flush=True)
    except Exception:
        print("Error creating vector store:", flush=True)
        traceback.print_exc()
        raise

def get_conversational_chain():
    """
    Build the QA chain using Groq LLM (unchanged). Swap llm creation if you want Google generative model instead.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in the provided context, please answer with "Sorry, I don't know the answer to that question."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    # llm = ChatGroq(
    #     groq_api_key=GROQ_API_KEY,
    #     model_name=GROQ_MODEL
    # )
    if GEN_MODEL and os.getenv("GOOGLE_API_KEY"):
        print("Using Google Generative model:", GEN_MODEL, flush=True)
        llm = ChatGoogleGenerativeAI(model=GEN_MODEL)
    else:
        print("Using Groq model:", GROQ_MODEL, flush=True)
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question: str, k: int = 5) -> str:
    """
    Run similarity search then query the chain. Prefer local FAISS if present; fallback to Pinecone if configured.
    Returns a string answer (never a raw dict).
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

        # prefer local FAISS index if available
        if os.path.exists("faiss_index"):
            try:
                db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = db.similarity_search(user_question, k=k)
                print("Used local FAISS for similarity search, found:", len(docs), flush=True)
            except Exception:
                print("Failed to load/use local FAISS index, falling back to Pinecone if available", flush=True)
                docs = []
        else:
            docs = []

        # if no docs and Pinecone configured, try Pinecone
        if not docs and PINECONE_API_KEY:
            try:
                print("Using Pinecone vector store for similarity search", flush=True)
                vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
                docs = vector_store.similarity_search(user_question, k=k, namespace=identity)
                print("Pinecone returned docs:", len(docs), flush=True)
            except Exception:
                print("Pinecone similarity search failed:", flush=True)
                traceback.print_exc()
                docs = []

        if not docs:
            return "No relevant documents found in vector store. Run ingestion to create an index."

        chain = get_conversational_chain()
        raw_resp = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        print("Raw chain response:", raw_resp, flush=True)

        # normalize response to string
        if isinstance(raw_resp, dict):
            for key in ("output_text", "text", "answer", "response"):
                if key in raw_resp and raw_resp[key]:
                    return raw_resp[key]
            # fallback: join any string-like values
            return str(raw_resp)
        return str(raw_resp)
    except Exception:
        print("Error in user_input:", flush=True)
        traceback.print_exc()
        return "Internal error while generating response. Check server logs."

# CLI runner for manual testing
if __name__ == "__main__":
    try:
        PDF_PATH = os.getenv("PDF_PATH", "uploads/Part1.pdf")
        QUESTION = os.getenv("QUESTION", "Give a short summary of the document.")
        print("Starting CLI run", flush=True)
        text = get_pdf_text(PDF_PATH)
        if not text.strip():
            print("No text extracted from PDF. Exiting.", flush=True)
            raise SystemExit(1)

        chunks = get_text_chunks(text)
        # create or overwrite local FAISS index
        get_vector_store(chunks)

        # perform similarity search and get answer
        answer = user_input(QUESTION)
        print("\n=== ANSWER ===\n", flush=True)
        print(answer, flush=True)
        print("\n=== END ===", flush=True)
    except Exception:
        print("Unhandled error in CLI run:", flush=True)
        traceback.print_exc()
# ...existing code...