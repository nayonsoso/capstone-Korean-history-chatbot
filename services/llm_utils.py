import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

# 모델·토크나이저 로드
logger.info(f"▶ LLM 모델 로드 시작: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
logger.info("▶ LLM 모델 및 토크나이저 로드 완료")

def call_llm_lg_ai(system_prompt: str, user_prompt: str, max_new_tokens: int, do_sample: bool) -> str:
    # 메시지 구성
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]

    # 토크나이징
    raw_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # 3) 토큰 형태 로깅 (딕셔너리 vs Tensor 분기)
    if isinstance(raw_inputs, dict):
        ids = raw_inputs["input_ids"]
    else:
        ids = raw_inputs
    batch_size, seq_len = ids.shape
    logger.debug(f"  • 배치 크기={batch_size}, 시퀀스 길이={seq_len}")

    # 디바이스 결정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if isinstance(raw_inputs, dict):
        # dict 형태라면 키별로 .to(device)
        inputs = {k: v.to(device) for k, v in raw_inputs.items()}
    else:
        # Tensor 하나라면 input_ids 키를 만들어서 dict 형태로 변환
        inputs = {"input_ids": raw_inputs.to(device)}
    logger.debug(f"사용 장치: {device}")

    # 생성
    output = model.generate(
        **inputs,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample
    )
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"생성된 응답: {result}")
    return result

def call_llm_chat_gpt(system_prompt: str, user_prompt: str, max_new_tokens: int, mode_l: str="gpt-4o-mini") -> str:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

    my_key = "<KEY>"

    client = OpenAI(api_key=my_key)
    messages = [
        ChatCompletionSystemMessageParam(content=system_prompt, role="system"),
        ChatCompletionUserMessageParam(content=user_prompt, role="user"),
    ]

    response = client.chat.completions.create(
        model=mode_l,
        messages=messages,
        temperature=0.7,
        max_tokens=max_new_tokens,
    )
    print(response.choices[0].message.content)

    content = response.choices[0].message.content
    logger.info(f"생성된 응답: {content}")
    return content
