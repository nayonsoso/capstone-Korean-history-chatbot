import os
import glob
import logging
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# 전역 설정
COLLECTION_NAME = "k-history"
EMBED_MODEL_NAME = "nlpai-lab/KoE5"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# Chroma 클라이언트 및 임베딩 모델 준비
client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings(anonymized_telemetry=False, allow_reset=False)
)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def init_chroma():
    """애플리케이션 시작 시 ChromaDB에 컬렉션을 초기화"""
    logger.info(f"▶ ChromaDB 초기화 시작: '{COLLECTION_NAME}' 컬렉션 확인")
    try:
        client.get_collection(name=COLLECTION_NAME)
        logger.info(f"✔ 컬렉션 '{COLLECTION_NAME}' 이미 존재 — 초기화 스킵")
    except NotFoundError:
        logger.info(f"✚ 컬렉션 '{COLLECTION_NAME}' 미발견 — 새로 생성")
        collection = client.create_collection(name=COLLECTION_NAME)

        # data/ 폴더 내 모든 .txt 파일을 읽어서 docs 리스트에 저장
        txt_files = glob.glob("data/*.txt")
        docs: List[str] = []
        metadatas: List[Dict[str, str]] = []

        logger.info(f"  • 로드할 텍스트 파일 개수: {len(txt_files)}개")
        for path in txt_files:
            filename = os.path.basename(path)
            logger.info(f"    – 파일 읽기 시작: {filename}")
            with open(path, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            docs.extend(lines)
            metadatas.extend([{"source": filename}] * len(lines))
            logger.info(f"    – '{filename}'에서 {len(lines)}개 문장 로드 완료")

        logger.info(f"  • 총 문장 수: {len(docs)}개")
        logger.info(f"  • 총 메타데이터 수: {len(metadatas)}개")

        # 임베딩
        logger.info("  • 임베딩 생성 중…")
        embs = embed_model.encode(docs).tolist()
        logger.info("  • 임베딩 생성 완료")

        # 컬렉션에 추가
        ids = [str(i) for i in range(len(docs))] # 각 문서에 대한 고유 ID 생성

        logger.info(f"  • ChromaDB에 {len(docs)}개 문서 저장 중…")
        collection.add(
            embeddings=embs,
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )
        logger.info("  • 문서 저장 완료")

    logger.info("▶ ChromaDB 초기화 완료")

def find_k_docs(query: str, k: int = 5) -> dict:
    """주어진 쿼리에 대해 상위 k개의 문서를 검색"""
    logger.info(f"▶ ChromaDB에서 '{query}'에 대한 상위 {k}개 문서 검색")

    # 질문 임베딩
    q_emb = embed_model.encode([query]).tolist()[0]

    collection = client.get_collection(name=COLLECTION_NAME)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )
    # 문서 내용 출력
    docs_found = results['documents'][0]
    metadatas_found = results['metadatas'][0]
    for i, (doc, meta) in enumerate(zip(docs_found, metadatas_found)):
        # meta 딕셔너리에서 'source' 키로 파일명을 가져옴
        source_file = meta.get('source', '알 수 없음')
        logger.info(f" 문서 {i+1} (출처: {source_file}): {doc}")

    logger.info(f"✔ 검색 완료: {len(results['ids'][0])}개 문서")
    return results

def is_similar(doc1: str, doc2: str, threshold: float) -> bool:
    """두 문서 간의 유사도를 계산하고, threshold 이상인지 판단"""
    logger.info(f"▶ '{doc1}'와 '{doc2}' 간 유사도 계산")

    # 쿼리 임베딩
    doc1_emb = embed_model.encode([doc1], convert_to_tensor=True).tolist()[0]
    doc2_emb = embed_model.encode([doc2], convert_to_tensor=True).tolist()[0]

    # 코사인 유사도 계산
    from sentence_transformers.util import cos_sim
    similarity_tensor = cos_sim(doc1_emb, doc2_emb)
    similarity = similarity_tensor.item()
    logger.info(f"✔ 유사도 계산 완료: {similarity:.4f}")

    is_similar_ = similarity >= threshold
    logger.info(f"  • 유사도 임계값 비교: {'유사' if is_similar else '비유사'}")
    return is_similar_