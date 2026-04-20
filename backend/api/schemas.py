from pydantic import BaseModel

class dbModel(BaseModel):
    user: str
    password: str
    host: str
    port: str
    database: str
    
class ChatRequest(BaseModel):
    user_query: str
    
class UrlModel(BaseModel):
    url: str