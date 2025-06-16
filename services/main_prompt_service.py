import logging
from typing import List
import json

from exception_handler import InternalServerException
from schemas import ServiceResponse, SummaryResponse, ServiceTextResponse, SummaryTextResponse

from services.llm_utils import call_llm_chat_gpt

logger = logging.getLogger(__name__)

max_tokens = 600

service_system_prompt = (
    """ 
        당신은 한국사를 알려주는 친구이다.
        주어지는 '문서'를 기반으로 질문에 답하라.
        만약, 주어진 문서에서 답을 찾을 수 없거나, 문서의 내용과 모순되는 질문을 한다면 'no'라고 짧게 응답하라.
        그게 아니라면, 질문에 대해 한번에 답변하지 않고 단계적으로 응답하려 한다. 응답은 크기가 3인 JSON 배열이다.
        각 요소는 다음 필드를 가져야 한다. 
        {
            "index": integer,  # 0부터 시작
            "summary": string, # 해당 답변 전체를 간략히 요약한 것
            "question": string, # 이전 hints를 바탕으로, 이전 질문의 정답이 무엇인지를 구체적으로 말한다. 그리고 추가 질문을 한다.
            "hints": string array # question의 대한 정답을 키워드 형태로 제시
        }
        추가로 주어지는 '문서'에서 관련된 내용을 찾아 답하면 좋다.

        예시 질문은 "이자겸의 난은 조선시대에 발생했어??"이다. 실제 답변은 예시보다 더 길게 답해야한다.
        [{
            "index": 0,
            "summary": "이자겸의 난이 언제 발생했는지 알아보자!"
            "question": "좋은 질문이야. 먼저, 이자겸의 난’이 일어난 시기를 떠올려 볼까?"
            "hints": ["고려시대", "무신정권"]
        }, {
            "index": 1,
            "summary": "이자겸의 난 발생 배경에 대해 알아보자!"
            "question": "맞아. 더 정확히는 고려시대 인종 말년에 발생했어. 그렇다면 이자겸의 난은 왜 발생했을까? 이자겸이라는 인물에 대해 생각해볼까?"
            "hints": ["권문세족", "왕실과 혼인"]
        }, {
            "index": 2,
            "summary": "이자겸의 난의 결과에 대해 알아보자!"
            "question": "잘했어. 이자겸은 권문세족의 일원으로, 왕실과의 혼인 관계를 통해 권력을 강화했어. 그 틈을 틈타 반란을 꾀한 것이 바로 이자겸의 난이야. 그렇다면 이자겸의 난은 결국 어떻게 끝났을까?"
            "hints": ["실패", "귀족사회의 동요"]
        }]
        제출 전에 반드시 정해진 JSON 형식이 맞는지 검토해야한다.
    """
)

service_system_prompt_tmp = (
    """ 
        당신은 한국사를 알려주는 친구이다. 질문에 대해 한번에 답변하지 않고 단계적으로 응답하려 한다. 응답은 크기가 3인 JSON 배열이다.
        각 요소는 다음 필드를 가져야 한다. 
        {
            "index": integer,  # 0부터 시작
            "summary": string, # 해당 답변 전체를 간략히 요약한 것
            "question": string, # 이전 hints를 바탕으로, 이전 질문의 정답이 무엇인지를 구체적으로 말한다. 그리고 추가 질문을 한다.
            "hints": string array # question의 대한 정답을 키워드 형태로 제시
        }
        추가로 주어지는 '문서'에서 관련된 내용을 찾아 답하면 좋다.

        예시 질문은 "이자겸의 난은 조선시대에 발생했어??"이다. 실제 답변은 예시보다 더 길게 답해야한다.
        [{
            "index": 0,
            "summary": "이자겸의 난이 언제 발생했는지 알아보자!"
            "question": "좋은 질문이야. 먼저, 이자겸의 난’이 일어난 시기를 떠올려 볼까?"
            "hints": ["고려시대", "무신정권"]
        }, {
            "index": 1,
            "summary": "이자겸의 난 발생 배경에 대해 알아보자!"
            "question": "맞아. 더 정확히는 고려시대 인종 말년에 발생했어. 그렇다면 이자겸의 난은 왜 발생했을까? 이자겸이라는 인물에 대해 생각해볼까?"
            "hints": ["권문세족", "왕실과 혼인"]
        }, {
            "index": 2,
            "summary": "이자겸의 난의 결과에 대해 알아보자!"
            "question": "잘했어. 이자겸은 권문세족의 일원으로, 왕실과의 혼인 관계를 통해 권력을 강화했어. 그 틈을 틈타 반란을 꾀한 것이 바로 이자겸의 난이야. 그렇다면 이자겸의 난은 결국 어떻게 끝났을까?"
            "hints": ["실패", "귀족사회의 동요"]
        }]
        제출 전에 반드시 정해진 JSON 형식이 맞는지 검토해야한다.
    """
)


summary_system_prompt = (
    """
        당신은 한국사를 알려주는 친구이다. 주어진 json 배열을 보고, 어떤 내용의 답변이 있었는지 요약하는 하나의 json을 만들어야 한다.
        {
            "questionSummary": string, # 질문을 간단히 요약한 것
            "responseSummary": string, # 응답을 간단히 요약한 것
            "thoughtProcess": string array, # 답변 시 활용한 단계적 사고 과정
            "keywords": string array # 답변 시 활용한 힌트를 간단히 요약한 키워드 목록
        }
        예시는 다음과 같다.
        {
            "questionSummary": "고려시대 인종 말년에 발생한 반란인, 이자겸의 난에 대해 알아봤어!",
            "responseSummary": "이자겸은 권문세족으로 왕실과 혼인 관계를 통해 권력을 강화했어. \n 이자겸의 난은 결국 실패로 끝났지만, 귀족 사회에 동요를 일으켰고 정치기강은 더욱 문란하게 했어. \n 이자겸의 난은 고려시대 권문세족의 권력 구조와 왕실 간의 관계를 이해하는 데 중요한 사건이야.",
            "thoughtProcess": [
                "이자겸의 난의 시기와 배경을 이해한다.",
                "이자겸이라는 인물의 역할과 권력 구조를 파악한다.",
                "이자겸의 난의 결과와 그로 인한 사회적 변화에 대해 생각한다."
            ],
            "keywords": ["고려시대 인조 말년", "권문세족", "왕실 외척", "반란", "실패"]
        }
        제출 전에 반드시 정해진 JSON 형식이 맞는지 검토해야한다.
    """
)

def generate_service_responses(question: str, k_docs: list, model_name: str="gpt-4o-mini") -> List[ServiceResponse]:
    from json import JSONDecodeError
    import json
    import re
    user_prompt = f"사용자 질문: {question}\n 문서: {json.dumps(k_docs, ensure_ascii=False)}\n"

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        response = call_llm_chat_gpt(service_system_prompt, user_prompt, max_tokens)
        if response == "no":
            logger.info("LLM 응답: 'no' - 한국사 관련 질문이 아님")
            return []

        raw = response.lstrip('\ufeff').strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw, count=1)
            raw = re.sub(r"\n?```$", "", raw, count=1).strip()

        # 최상위가 {…}, {…} 형태면 배열로 감싸기
        if raw and raw[0] != "[":
            raw = f"[{raw}]"

        try:
            data = json.loads(raw)
            break  # 파싱 성공 시 루프 탈출
        except JSONDecodeError:
            logger.error("LLM 응답 JSON 파싱 실패 (시도 %d/%d): %s", attempt, max_retries, raw)
            if attempt == max_retries:
                raise InternalServerException("LLM 응답이 Json 으로 변환되지 않습니다.")
            # 파싱 실패 시 같은 프롬프트로 재요청

    # data에 있는 것은 ServiceTextResponse에 담고,
    # 그것을 모두 ServiceResponse로 감싸서 반환
    services = [ServiceTextResponse(**item) for item in data]
    return [ServiceResponse(text=s) for s in services]

def generate_service_responses_tmp(question: str, k_docs: list, model_name: str="gpt-4o-mini") -> List[ServiceResponse]:
    from json import JSONDecodeError
    import json
    import re
    user_prompt = f"사용자 질문: {question}"

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        response = call_llm_chat_gpt(service_system_prompt_tmp, user_prompt, max_tokens)
        if response == "no":
            logger.info("LLM 응답: 'no' - 한국사 관련 질문이 아님")
            return []

        raw = response.lstrip('\ufeff').strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw, count=1)
            raw = re.sub(r"\n?```$", "", raw, count=1).strip()

        # 최상위가 {…}, {…} 형태면 배열로 감싸기
        if raw and raw[0] != "[":
            raw = f"[{raw}]"

        try:
            data = json.loads(raw)
            break  # 파싱 성공 시 루프 탈출
        except JSONDecodeError:
            logger.error("LLM 응답 JSON 파싱 실패 (시도 %d/%d): %s", attempt, max_retries, raw)
            if attempt == max_retries:
                return []
            # 파싱 실패 시 같은 프롬프트로 재요청

    # data에 있는 것은 ServiceTextResponse에 담고,
    # 그것을 모두 ServiceResponse로 감싸서 반환
    services = [ServiceTextResponse(**item) for item in data]
    return [ServiceResponse(text=s) for s in services]



def generate_summary_response(services: List[ServiceTextResponse]) -> SummaryResponse:
    import json
    from json import JSONDecodeError

    # 서비스 응답 리스트를 JSON으로 직렬화
    payload = json.dumps([r.model_dump() for r in services], ensure_ascii=False)
    user_prompt = f"json 배열:\n{payload}\n"

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        response = call_llm_chat_gpt(summary_system_prompt, user_prompt, max_tokens)

        try:
            result = json.loads(response)
            break  # 파싱 성공 시 루프 탈출
        except JSONDecodeError:
            logger.error("LLM 요약 응답 JSON 파싱 실패 (시도 %d/%d): %s", attempt, max_retries, response)
            if attempt == max_retries:
                raise InternalServerException("LLM 요약 응답이 Json 으로 변환되지 않습니다.")
            # 파싱 실패 시 같은 프롬프트로 재요청

    summary_text = SummaryTextResponse(**result)
    return SummaryResponse(text=summary_text)

def generate_summary_response_test(services: List[ServiceResponse]) -> SummaryResponse:
    import json
    from json import JSONDecodeError

    # 서비스 응답 리스트를 JSON으로 직렬화
    payload = json.dumps([r.model_dump() for r in services], ensure_ascii=False)
    user_prompt = f"json 배열:\n{payload}\n"

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        response = call_llm_chat_gpt(summary_system_prompt, user_prompt, max_tokens)

        try:
            result = json.loads(response)
            break  # 파싱 성공 시 루프 탈출
        except JSONDecodeError:
            logger.error("LLM 요약 응답 JSON 파싱 실패 (시도 %d/%d): %s", attempt, max_retries, response)
            if attempt == max_retries:
                raise InternalServerException("LLM 요약 응답이 Json 으로 변환되지 않습니다.")
            # 파싱 실패 시 같은 프롬프트로 재요청

    summary_text = SummaryTextResponse(**result)
    return SummaryResponse(text=summary_text)
