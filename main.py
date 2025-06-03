import logging

from fastapi.middleware.cors import CORSMiddleware
from exception_handler import BadRequestException, InternalServerException
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


@app.exception_handler(BadRequestException)
async def bad_request_exception_handler(request: Request, exc: BadRequestException):
    return JSONResponse(
        status_code=400,
        content={"number": 400, "message": exc.message}
    )


@app.exception_handler(InternalServerException)
async def internal_server_exception_handler(request: Request, exc: InternalServerException):
    return JSONResponse(
        status_code=500,
        content={"number": 500, "message": exc.message}
    )

# ----------------------
# 라우터 등록
# ----------------------
app.include_router(query_router)
app.include_router(rag_router)
app.include_router(llm_router)

app.include_router(main_router)

allow_origins = ["localhost:5173", "http://localhost:5173/step"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[allow_origins],  # 모든 도메인 허용
    allow_credentials=True,  # 쿠키, 인증 헤더 허용
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

logger.info("FastAPI 애플리케이션 초기화 완료")
