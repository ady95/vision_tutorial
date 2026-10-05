"""10장 공통 — vLLM 서버(OpenAI 호환 API)에 이미지와 질문을 보내는 함수

서버 실행 예 (별도 가상환경에 vllm 설치, 10-2 참고):
    vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 16384
"""
import base64
import io
import re
import time

from openai import OpenAI
from PIL import Image

BASE_URL = "http://localhost:8000/v1"
_client = {}


def client(base_url=BASE_URL):
    if base_url not in _client:
        _client[base_url] = OpenAI(base_url=base_url, api_key="EMPTY")             # 로컬 서버라 키는 아무 값
    return _client[base_url]


def model_name(base_url=BASE_URL):
    """서버에 올라간 모델 이름 (예: Qwen/Qwen3.5-4B)"""
    return client(base_url).models.list().data[0].id


def encode(image, max_side=None):
    """PIL 이미지 → data URL (JPEG). max_side를 주면 긴 변을 그 크기로 줄인다"""
    image = image.convert("RGB")
    if max_side and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(prompt, images=(), think=False, max_tokens=1024, base_url=BASE_URL, max_side=None):
    """이미지 여러 장 + 질문 → {"text", "thinking", "seconds", "prompt_tokens", "completion_tokens"}
    think=False면 thinking(답하기 전에 길게 생각하는 단계)을 끈다"""
    content = [{"type": "image_url", "image_url": {"url": encode(im, max_side)}} for im in images]
    content.append({"type": "text", "text": prompt})
    t = time.perf_counter()
    r = client(base_url).chat.completions.create(
        model=model_name(base_url), messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens, temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": think}})
    sec = time.perf_counter() - t
    text = r.choices[0].message.content or ""
    thinking = getattr(r.choices[0].message, "reasoning_content", None) or ""
    if "</think>" in text:                                                          # 생각과 답이 한 문자열로 온 경우
        thinking, text = text.split("</think>", 1)
        thinking = thinking.replace("<think>", "")
    return {"text": text.strip(), "thinking": thinking.strip(), "seconds": sec,
            "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens}


def first_int(text):
    """답에서 첫 번째 정수 (개수 세기 채점용, 없으면 None)"""
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group()) if m else None
