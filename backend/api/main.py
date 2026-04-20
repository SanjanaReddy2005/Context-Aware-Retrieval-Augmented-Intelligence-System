from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from mysql_chat import get_response, init_database
from groq_pdf_chat import get_pdf_text, get_text_chunks, get_vector_store, user_input
from website_chat import get_vectorstore_from_url, get_context_retriever_chain, get_conversational_rag_chain
from schemas import dbModel, ChatRequest, UrlModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for storing chat history, database connection info, and database
chat_history_sql: list = [AIMessage(content="Hello! I am a SQL Assistant. How can I help you?")]
database_data: dbModel = None
data = None
save_to = None
chat_history_wb: list = [AIMessage(content="Hello! I am a Website Assistant. How can I help you?")]

@app.post("/api/connect-to-db")
async def connect_to_db(db: dbModel):
    global database_data, data
    try:
        database_data = db
        data = init_database(db)
        return {"message": "Connection Established! You can now start querying the database.."}
    except Exception as e:
        return {"status": "Error connecting to database", "error": str(e)}
    
@app.post("/api/get-sql-response")
async def get_response_api(request: ChatRequest):
    global chat_history_sql
    try:
        chat_history_sql.append(HumanMessage(content=request.user_query))
        response = get_response(request.user_query, data, chat_history_sql)
        print(response)
        chat_history_sql.append(AIMessage(content=response))
        return {"response": response}
    except Exception as e:
        return {"status": "Error getting response", "error": str(e)}

@app.post("/api/upload-file")
async def upload_file(file: UploadFile):
    global save_to
    try:
        pdf = await file.read()
        save_to = f"uploads/{file.filename}"
        with open(save_to, "wb") as f:
            f.write(pdf)
            text = get_pdf_text(save_to)
            text_chunks = get_text_chunks(text)
            get_vector_store(text_chunks)
        return {"message": "File uploaded successfully! Start querying the document.."}
    except Exception as e:
        return {"status": "Error uploading file", "error": str(e)}

@app.post("/api/get-pdf-response")
def get_ai_response(request: ChatRequest):
    if save_to is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    try:
        response = user_input(request.user_query)
        return {"response": response}
    except Exception as e:
        return {"status": "Error getting response", "error": str(e)}
    
@app.post("/api/scrape-website")
async def scrape_website(url: UrlModel):
    try:
        get_vectorstore_from_url(url.url)
        return {"message": "Website scraped successfully! Start querying the website.."}
    except Exception as e:
        return {"status": "Error scraping website", "error": str(e)}
    
@app.post("/api/get-website-response")
def get_website_response(request: ChatRequest):
    global chat_history_wb
    try:
        chat_history_wb.append(HumanMessage(content=request.user_query))
        retriever_chain = get_context_retriever_chain()
        rag_chain = get_conversational_rag_chain(retriever_chain)
        response = rag_chain.invoke(
            {"input": request.user_query, "chat_history": chat_history_wb}
        )
        chat_history_wb.append(AIMessage(content=response["answer"]))
        return {"response": response["answer"]}
    except Exception as e:
        return {"status": "Error getting response", "error": str(e)}