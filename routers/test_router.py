from fastapi import Request, Response, APIRouter
import uuid
import logging

from starlette.responses import JSONResponse

from exception_handler import BadRequestException
from schemas import QuestionRequest
from services.chroma_service import find_k_documents, is_answer_related_to_hints
from services.main_prompt_service import generate_service_responses, generate_summary_response, \
    generate_summary_response_test

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/test")
async def process_question(request: Request, response: Response,) -> JSONResponse:
    # 1) 파싱
    qreq = await convert_request(request)
    question = qreq.question

    k_docs = find_k_documents(question)
    response_list = generate_service_responses(question, k_docs)
    if not response_list:
        logger.info("✖ 관련된 답변을 찾을 수 없음")
        raise BadRequestException("한국사와 관련된 질문을 해줘!")

    summary = generate_summary_response_test(response_list)
    return JSONResponse(
        content={
            "responses": [r.model_dump() for r in response_list],
            "summary": summary.model_dump(),
        }
    )

async def convert_request(request: Request) -> QuestionRequest:
    if request.url.path == "/test" and request.method.upper() == "POST":
        try:
            payload = await request.json()
            return QuestionRequest(**payload)
        except Exception as ex:
            logger.info("✖ /question 경로에 형식에 맞지 않는 요청")
    raise BadRequestException("잘못된 요청 형식이야.")
