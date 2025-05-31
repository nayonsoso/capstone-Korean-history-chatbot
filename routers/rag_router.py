import logging
from fastapi import APIRouter, HTTPException
from schemas import RagRequest, RagResponse
from services.chroma_utils import client, COLLECTION_NAME, embed_model
from services.llm_utils import call_llm_lg_ai

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/llm/lg-ai/rag", response_model=RagResponse)
async def rag(req: RagRequest):
    """
    RAG endpoint: VectorDB에서 관련 문서를 검색하고,
    한국사 단계적 사고에 맞춰 세 단계로 나누어 답변을 생성합니다.
    """
    try:
        logger.info(f"▶ /rag 요청: '{req.prompt}', top {req.k}")
        # 1) 질문 임베딩 생성
        q_emb = embed_model.encode([req.prompt]).tolist()[0]

        # 2) ChromaDB에서 상위 k개 문서 검색
        collection = client.get_collection(name=COLLECTION_NAME)
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=req.k
        )
        logger.debug(f"검색 결과 IDs: {results.get('ids')}")

        # 3) 검색된 문서 결합
        docs = results.get('documents', [[]])[0]
        context = "\n\n".join(docs)

        # 4) LLM 프롬프트 구성
        system_prompt = (
            "너는 한국사를 알려주는 친구야. 친구가 단계별로 점진적인 사고를 할 수 있도록 도와줘야해. "
            "내가 주는 문서를 기반으로 답변을 하되, 단계적 사고를 할 수 있도록 3개의 대화로 끊어서 제공해줘"
        )
        full_prompt = f"{system_prompt}\n\n[문서]\n{context}\n\n[질문]\n{req.prompt}"
        logger.debug(f"전달된 프롬프트: {full_prompt}")

        # 5) LLM에 프롬프트 전달하여 생성
        response_text = call_llm_lg_ai(
            prompt=full_prompt,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample
        )
        logger.info("✔ /rag 완료")
        return RagResponse(response=response_text)

    except Exception as e:
        logger.exception("✖ /rag 처리 중 오류")
        raise HTTPException(status_code=500, detail=str(e))
