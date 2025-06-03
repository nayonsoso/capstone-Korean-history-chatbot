import logging
import glob
from pathlib import Path
from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer
from chromadb.errors import NotFoundError

from services.chroma_utils import COLLECTION_NAME

router = APIRouter()

# ─── 예시상정: 이미 init_chroma()에서 client와 embed_model이 전역으로 정의되어 있다고 가정 ───
# from chromadb import HttpClient
# client = HttpClient(host="localhost", port=8000, settings=Settings(...))
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# COLLECTION_NAME = "k-history"
# ─────────────────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


@router.get("/chroma/docs")
async def read_all_chroma_docs():
    """
    ChromaDB에 저장된 모든 문서(=청크) ID와 본문을 반환합니다.
    - 컬렉션이 없으면 404 에러를 내보냅니다.
    - n_results를 collection.count()에 맞춰서 모든 문서를 가져옵니다.
    """
    try:
        from services.chroma_utils import client
        collection = client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Collection '{COLLECTION_NAME}' not found")

    # 컬렉션 안에 들어 있는 문서 총 개수 파악
    total_docs = collection.count()

    # 실제 저장된 모든 문서를 꺼내기 위해 n_results=total_docs 설정
    # include 인자를 생략하면, 기본적으로 ids, documents, embeddings, metadatas 전부 가져옵니다.
    data = collection.get()

    # Chroma에서 반환하는 data 구조 예시:
    # {
    #   "ids":        ["0", "1", "2", ...],
    #   "documents":  ["첫 번째 청크 내용", "두 번째 청크 내용", ...],
    #   "embeddings": [[...], [...], ...],
    #   "metadatas":  [ {...}, {...}, ... ]  # (만약 metadata를 넣었다면)
    # }
    #
    # 여기서는 사용자가 보고 싶어 하는 “문서(청크) 내용”만 뽑아서 리턴합니다.

    docs = []
    for idx, doc_text in zip(data["ids"], data["documents"]):
        docs.append({
            "id": idx,
            "document": doc_text
        })

    return {"documents": docs}
