"""11-3. VLM 서빙 비교 — Transformers, vLLM, llama.cpp

같은 질문("자세히 설명하라", 출력 256토큰)으로 잰다. 서버가 같은 이미지의 결과를 재사용(prefix cache)하지 않도록
측정마다 다른 이미지를 쓴다: 워밍업 coco_cats.jpg, 한 개 측정 bus.jpg, 동시 요청은 COCO 사진을 요청마다 다르게.
- 첫 토큰까지 시간(TTFT): 이미지를 읽고(prefill) 첫 글자를 내기까지
- 생성 속도: 첫 토큰 이후 초당 토큰
- 동시 요청: 요청 1·4·8개를 한꺼번에 보냈을 때 전체 초당 토큰 (서버만)
- GPU 메모리: nvidia-smi로 본 사용량

실행 (저장소 루트에서):
    python ch11_deploy/11_3_serving_bench.py --base-url http://localhost:8000/v1      # vLLM 또는 llama-server
    python ch11_deploy/11_3_serving_bench.py --transformers Qwen/Qwen3.5-4B          # 서버 없이
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ch10_vlm"))
from vlm import client, encode, model_name  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--transformers", default="", help="서버 없이 transformers로 불러올 모델 이름")
args = ap.parse_args()
img = Image.open(ROOT / "data" / "images" / "bus.jpg")
PROMPT, N_OUT = "Describe this image in detail.", 256


def gpu_mem():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return " / ".join(f"{int(v) / 1024:.1f}" for v in out.stdout.split()) + " GiB"


if args.transformers:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t = time.perf_counter()
    proc = AutoProcessor.from_pretrained(args.transformers)
    model = AutoModelForImageTextToText.from_pretrained(args.transformers, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"{args.transformers} (transformers BF16): 불러오기 {time.perf_counter() - t:.0f}초")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                      return_tensors="pt", enable_thinking=False).to("cuda")
    times = {}
    for n in [1, 1, N_OUT]:                                                   # 첫 번째는 워밍업
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False)
        torch.cuda.synchronize()
        times[n] = time.perf_counter() - t
    print(f"  TTFT {times[1]:.2f}초 | 생성 {(N_OUT - 1) / (times[N_OUT] - times[1]):.1f} 토큰/초 | GPU 메모리 {gpu_mem()}")
    raise SystemExit

c, m = client(args.base_url), model_name(args.base_url)
coco = sorted((ROOT / "data" / "datasets" / "coco_val500" / "images").glob("*.jpg"))


def one(image):
    """스트리밍으로 받아 첫 토큰 시각과 끝 시각을 잰다"""
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": encode(image)}}, {"type": "text", "text": PROMPT}]}]
    t0, first, n = time.perf_counter(), None, 0
    stream = c.chat.completions.create(model=m, messages=messages, max_tokens=N_OUT, temperature=0.0, stream=True,
                                       extra_body={"chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True})
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            first = first or time.perf_counter()
            n += 1
    end = time.perf_counter()
    return first - t0, n, end - first, end - t0


print(f"{m} ({args.base_url})")
one(Image.open(ROOT / "data" / "images" / "coco_cats.jpg"))                  # 워밍업
ttft, n, gen, _ = one(img)
print(f"  TTFT {ttft:.2f}초 | 생성 {(n - 1) / gen:.1f} 토큰/초 ({n}조각) | GPU 메모리 {gpu_mem()}")
for k, start in [(1, 0), (4, 1), (8, 5)]:                                   # 요청마다 다른 COCO 사진
    t = time.perf_counter()
    with ThreadPoolExecutor(k) as ex:
        res = list(ex.map(lambda p: one(Image.open(p)), coco[start:start + k]))
    sec = time.perf_counter() - t
    print(f"  동시 {k}개: 전체 {sum(r[1] for r in res) / sec:.1f} 토큰/초, 요청당 평균 {sum(r[3] for r in res) / k:.1f}초")
