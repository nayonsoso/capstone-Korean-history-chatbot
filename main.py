import logging

from routers.main_router import router as main_router
from routers.llm_router import router as llm_router
from routers.chroma_router import router as query_router
from routers.rag_router import router as rag_router
from services.chroma_utils import init_chroma
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ----------------------
# 1) 로깅 설정
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ----------------------
# 2) FastAPI 앱 초기화
# ----------------------
app = FastAPI(on_startup=[init_chroma])

# ----------------------
# 라우터 등록
# ----------------------
app.include_router(query_router)
app.include_router(rag_router)
app.include_router(llm_router)

app.include_router(main_router)

logger.info("FastAPI 애플리케이션 초기화 완료")
