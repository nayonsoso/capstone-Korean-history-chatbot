import logging
from typing import List
import json

from exception_handler import InternalServerException
from schemas import ServiceResponse, SummaryResponse, ServiceTextResponse, SummaryTextResponse

from services.llm_utils import call_llm_chat_gpt

logger = logging.getLogger(__name__)

max_tokens = 600

service_system_prompt = (
    """ 당신은 친절하게 한국사를 알려주는 친구이다. 질문에 대해, 한번에 답변하지 않고 단계적으로 질문, 응답하려 한다. JSON 배열 형식으로, 배열의 크기는 3이다.
        각 요소는 다음 필드를 가져야 한다. 
        {
            "index": integer,  # 0부터 시작
            "summary": string, # 해당 답변 전체를 간략히 요약한 것
            "question": string, # idx:0이라면, 사용자의 답변이 yes,no로 대답할 수 있는 질문일 때는 이에 대답한다. (맞아, 아니야 등) yes,no 로 대답할 수 있는게 아니라면 '좋은 질문이야'와 같이 칭찬한다. / idx:1,2라면, 사용자의 답변이 좀 더 구체적인 응답하고, 추가 질문을 한다.
            "hints": string array # 답변 시 활용할 힌트들
        }
        추가로 주어지는 '문서'에서 관련된 내용을 찾아 답하면 좋다.

        예시 질문은 "이자겸의 난은 조선시대에 발생했어??"이다.
        [{
            "index": 0,
            "summary": "이자겸의 난이 언제 발생했는지 알아보자!"
            "question": "조선시대에 발생한건 아니야. 힌트를 통해 이자겸의 난이 일어난 시기를 떠올려 볼까?"
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
        제출 전에 반드시 정해진 JSON 형식이 맞는지 검토해주라.
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
        제출 전에 반드시 정해진 JSON 형식이 맞는지 검토해주라.
    """
)

def generate_service_responses(question: str, k_docs: list) -> List[ServiceResponse]:
    # user_prompt = f"사용자 질문: {question}\n 문서:{json.dumps(k_docs, ensure_ascii=False)}\n"
    user_prompt = f"사용자 질문: {question}\n"
    response = call_llm_chat_gpt(service_system_prompt, user_prompt, max_tokens)

    raw = response.lstrip('\ufeff').strip()
    if raw.startswith("```"):
        import re
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw, count=1)
        raw = re.sub(r"\n?```$", "", raw, count=1).strip()

    # 최상위가 {…}, {…} 형태면 배열로 감싸기
    if raw and raw[0] != "[":
        raw = f"[{raw}]"

    from json import JSONDecodeError
    try:
        data = json.loads(response)
    except JSONDecodeError:
        logger.error("LLM 응답 JSON 파싱 실패: %s", response)
        raise InternalServerException("LLM 응답이 Json 으로 변환되지 않습니다.")

    # data에 있는 것은 ServiceTextResponse에 담고,
    # 그것을 모두 ServiceResponse로 감싸서 반환
    services = [ServiceTextResponse(**item) for item in data]
    return [ServiceResponse(text=s) for s in services]

def generate_summary_response(services: List[ServiceTextResponse]) -> SummaryResponse:
    # 먼저 리스트를 JSON으로 직렬화
    payload = json.dumps([r.model_dump() for r in services], ensure_ascii=False)
    user_prompt = f"json 배열:\n{payload}\n"
    response = call_llm_chat_gpt(summary_system_prompt, user_prompt, max_tokens)

    from json import JSONDecodeError
    try:
        result = json.loads(response)
    except JSONDecodeError:
        logger.error("LLM 요약 응답 JSON 파싱 실패: %s", response)
        raise InternalServerException("LLM 응답이 Json 으로 변환되지 않습니다.")

    # result를 SummaryTextResponse로 변환하고,
    # 그것을 SummaryResponse로 감싸서 반환
    summary_text = SummaryTextResponse(**result)
    return SummaryResponse(text=summary_text)
