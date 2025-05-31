import logging
from fastapi import APIRouter, HTTPException

from schemas import ChatRequest, ChatResponse
from services.llm_utils import call_llm_lg_ai, call_llm_chat_gpt

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/llm/lg-ai", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"▶ /llm/lg-ai 요청: {request.prompt}")
        text = call_llm_lg_ai(
            system_prompt="너는 한국사를 친절히 설명해주는 친구야. 사용자의 질문에 대해 단계적으로 답변해줘.", # todo: 단계적으로 실험 필요
            user_prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample
        )
        return ChatResponse(response=text)
    except Exception as e:
        logger.exception("✖ /chat 처리 중 오류")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/llm/chat-gpt", response_model=ChatResponse)
async def chat_gpt(request: ChatRequest):
    try:
        logger.info(f"▶ /llm/chat-gpt 요청: {request.prompt}")
        text = call_llm_chat_gpt(
            system_prompt="너는 한국사를 친절히 설명해주는 친구야. 사용자의 질문에 대해 단계적으로 답변해줘.", # todo: 단계적으로 실험 필요
            user_prompt=request.prompt,
            max_new_tokens=request.max_new_tokens
        )
        return ChatResponse(response=text)
    except Exception as e:
        logger.exception("✖ /chat-gpt 처리 중 오류")
        raise HTTPException(status_code=500, detail=str(e))