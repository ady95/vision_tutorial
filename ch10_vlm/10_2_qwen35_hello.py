"""10-2. Qwen3.5 시작하기 — 첫 추론

1) vLLM 서버(OpenAI 호환 API)로 — thinking 끔/켬, 시간·토큰
2) transformers로 직접 불러와서 (--transformers, GPU에 서버가 떠 있지 않을 때) — 메모리·시간

실행 (저장소 루트에서):
    vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 16384 --max-num-seqs 16   # 다른 터미널, vllm 가상환경
    python ch10_vlm/10_2_qwen35_hello.py
    python ch10_vlm/10_2_qwen35_hello.py --transformers Qwen/Qwen3.5-2B               # 서버 없이
"""
import argparse
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm import ask, model_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--transformers", default="", help="서버 없이 transformers로 불러올 모델 이름")
args = ap.parse_args()
img = Image.open(ROOT / "data" / "images" / "bus.jpg")
PROMPT = "이 사진에 무엇이 있는지 한국어 한 문장으로 설명하라."

if not args.transformers:
    # 1. vLLM 서버로
    print(f"서버 모델: {model_name(args.base_url)}")
    ask(PROMPT, [img], base_url=args.base_url)                                    # 워밍업
    for think in [False, True]:
        r = ask(PROMPT, [img], think=think, max_tokens=4096, base_url=args.base_url)
        print(f"\nthinking {'켬' if think else '끔'}: {r['seconds']:.2f}초 | 입력 {r['prompt_tokens']} 토큰 (이미지 포함) | "
              f"출력 {r['completion_tokens']} 토큰 | 초당 {r['completion_tokens'] / r['seconds']:.0f} 토큰")
        if think:
            print(f"  생각 ({len(r['thinking'])}자): {r['thinking'][:300]} ...")
        print(f"  답: {r['text']}")
if args.transformers:
    # 2. transformers로 직접 (서버 없이)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t = time.perf_counter()
    proc = AutoProcessor.from_pretrained(args.transformers)
    model = AutoModelForImageTextToText.from_pretrained(args.transformers, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"{args.transformers}: {type(model).__name__}, 파라미터 {sum(p.numel() for p in model.parameters()):,}, "
          f"불러오기 {time.perf_counter() - t:.1f}초, GPU 메모리 {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                      return_tensors="pt", enable_thinking=False).to(model.device)
    for i in range(2):                                                            # 두 번째가 워밍업 뒤의 시간
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t
    new = out[0, inputs["input_ids"].shape[1]:]
    print(f"입력 {inputs['input_ids'].shape[1]} 토큰, 출력 {len(new)} 토큰, {sec:.2f}초 (초당 {len(new) / sec:.0f} 토큰), "
          f"최대 GPU 메모리 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"답: {proc.decode(new, skip_special_tokens=True).strip()}")
