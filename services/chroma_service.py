import logging

from exception_handler import BadRequestException

logger = logging.getLogger(__name__)

from services.chroma_utils import find_k_docs, is_similar

def find_k_documents(question: str, k:int = 3, threshold:float = 0.2) -> list:
    logger.info(f"▶ 주제 관련성 검사 시작: 질문 - '{question}', K - {k}, 임계값 - {threshold}")

    # K개의 문서 검색
    k_docs = find_k_docs(question, k)

    documents = k_docs.get('documents', [[]])[0]
    if not documents:
        logger.info("✖ 관련 문서 없음")
        raise BadRequestException("한국사와 관련된 질문을 해줘!")

    # 유사도 평균 계산
    distances = k_docs.get('distances', [[]])[0]
    avg_similarity = sum(distances) / len(distances)
    logger.info(f"  • 평균 유사도: {avg_similarity:.4f}")
    if avg_similarity < threshold :
        logger.info("✖ 주제 관련성 부족")
        raise BadRequestException("한국사와 관련된 질문을 해줘!")

    return documents

def is_answer_related_to_hints(hints: list[str], additional_answer: str, threshold:float = 0.5) -> bool:
    joined_hints = " ".join(hints)
    logger.info(f"▶ 힌트 관련성 검사 시작: 힌트 - '{joined_hints}', 추가 답변 - '{additional_answer}', 임계값 - {threshold}")

    # 두 질문을 임베딩 한 값의 유사도 비교
    is_related = is_similar(joined_hints, additional_answer, threshold)
    logger.info(f"✔ 관련성 검사 완료: {'관련 있음' if is_related else '관련 없음'}")

    return is_related