from pydantic import BaseModel, Field
from typing import List, Literal


class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    do_sample: bool = False

class ChatResponse(BaseModel):
    response: str

class QueryRequest(BaseModel):
    prompt: str
    k: int = 5

class RagRequest(BaseModel):
    prompt: str
    k: int = 3
    max_new_tokens: int = 256
    do_sample: bool = False

class RagResponse(BaseModel):
    response: str

class QuestionRequest(BaseModel):
    question: str

class ServiceTextResponse(BaseModel):
    index: int
    summary: str
    question: str
    hints: List[str]

class ServiceResponse(BaseModel):
    type: Literal["service"] = "service"
    text: ServiceTextResponse

class SummaryTextResponse(BaseModel):
    questionSummary: str
    responseSummary: str
    thoughtProcess: List[str]
    keywords: List[str]

class SummaryResponse(BaseModel):
    type: Literal["summary"] = "summary"
    text: SummaryTextResponse

class ResponseWrapper(BaseModel):
    service: List[ServiceResponse]
    summary: SummaryResponse
