from fastapi import Request, Response, APIRouter
import uuid
import logging

from starlette.responses import JSONResponse

from exception_handler import BadRequestException
from schemas import QuestionRequest
from services.chroma_service import find_k_documents, is_answer_related_to_hints
from services.main_prompt_service import generate_service_responses, generate_summary_response

logger = logging.getLogger(__name__)
router = APIRouter()

SESSION_COOKIE_NAME = "session_id"
sessions = {}  # 세션 저장소에 저장될 내용 : {"session_id": {"count": 0, "response_list": []}}

@router.post("/question")
async def process_question(request: Request, response: Response,) -> JSONResponse:
    """
    1) Request → QuestionRequest
    2) 한국사 관련 질문인지 체크
    3) 세션 생성 또는 조회
    4) 카운트 증가 및 인덱스 계산
    5) ServiceResponse 리스트에서 차례대로 응답하다가, 다 쓴 뒤에는 SummaryResponse 를 반환하고 쿠키 삭제
    """

    # 1) 파싱
    qreq = await convert_request(request)
    question = qreq.question

    # 2) 세션 조회, 없다면 신규 세션 생성
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    new_session = False

    if not session_id or session_id not in sessions: # 첫 질문
        # chroma db에서 유사한 질문 검색, 없으면 예외
        k_docs = find_k_documents(question)
        response_list = generate_service_responses(question, k_docs)

        # 새로운 세션 생성
        session_id = str(uuid.uuid4())  # 세션 아이디 생성
        sessions[session_id] = {"count": 0, "response_list": response_list}
        new_session = True
        logger.info("✔ 새로운 세션 생성 : {}".format(session_id))
    else:
        logger.info(f"✔ 기존 세션 사용 : {session_id}")
        previous_count = sessions[session_id]["count"]
        previous_hints = sessions[session_id]["response_list"][previous_count - 1].text.hints
        # 이전 힌트와 관련된 질문인지 검사
        if not is_answer_related_to_hints(previous_hints, question):
            logger.info("✖ 이전 힌트와 관련 없는 질문")
            raise BadRequestException("이전 힌트와 관련된 대답을 해줘! 힌트로 주어지는 키워드들을 토대로 문장을 만들면, 네가 더 오래 기억할 수 있게 될거야.")

    # 4) 카운트 증가 및 인덱스 계산
    sessions[session_id]["count"] += 1
    count = sessions[session_id]["count"]
    logger.info(f"✔ 세션 카운트 [{session_id}]: {count}")

    idx = count - 1
    response_list = sessions[session_id]["response_list"]

    # 5) 아직 남은 ServiceResponse 가 있으면 하나 꺼내서 반환
    if idx < len(response_list):
        svc_resp = response_list[idx]
        json_resp = JSONResponse(svc_resp.dict())
        # 신규 세션일 때만 쿠키 설정
        if new_session:
            json_resp.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=session_id,
                httponly=True,
            )
        return json_resp

    logger.info("✔ 모든 단계를 완료")
    summary = generate_summary_response(response_list)
    json_resp = JSONResponse(content=summary.model_dump())
    json_resp.delete_cookie(SESSION_COOKIE_NAME)
    sessions.pop(session_id, None)
    return json_resp


async def convert_request(request: Request) -> QuestionRequest:
    if request.url.path == "/question" and request.method.upper() == "POST":
        try:
            payload = await request.json()
            return QuestionRequest(**payload)
        except Exception as ex:
            logger.info("✖ /question 경로에 형식에 맞지 않는 요청")
    raise BadRequestException("잘못된 요청 형식이야.")
