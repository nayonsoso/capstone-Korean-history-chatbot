import logging
from fastapi import APIRouter

from schemas import QueryRequest
from services.chroma_utils import client, COLLECTION_NAME, embed_model, find_k_docs

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/chroma-db/k-query")
async def query_chroma(req: QueryRequest):
    logger.info(f"▶ /query 요청: '{req.prompt}' 상위 {req.k}개 검색")
    results = find_k_docs(query=req.prompt, k=req.k)
    logger.info("✔ /query 완료")
    return {"results": results}
