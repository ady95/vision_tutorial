"""11-1. GPU 메모리와 정밀도 이해하기

1) 숫자 형식 — FP32 / FP16 / BF16의 범위와 정밀도, 넘치거나 뭉개지는 예
2) YOLO26s — 가중치와 Activation 메모리 (배치 크기 · FP32/FP16)
3) Qwen3.5 — 가중치, KV Cache, 선형 Attention 상태 (설정 파일로 계산 + transformers로 실제 측정)

실행 (저장소 루트에서, GPU가 비어 있을 때):
    python ch11_deploy/11_1_memory_precision.py
    python ch11_deploy/11_1_memory_precision.py --vlm Qwen/Qwen3.5-4B
"""
import argparse
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--vlm", default="Qwen/Qwen3.5-2B", help="실제로 불러와 측정할 VLM")
args = ap.parse_args()
GiB, MiB = 2**30, 2**20

# 1. 숫자 형식
print("형식      비트  가장 큰 수        1 다음의 수(eps)   유효 자릿수")
for name, dt in [("FP32", torch.float32), ("FP16", torch.float16), ("BF16", torch.bfloat16)]:
    f = torch.finfo(dt)
    print(f"{name:<8}{f.bits:>5}  {f.max:<17.4g} {f.eps:<18.3g} 약 {-torch.log10(torch.tensor(f.eps)).item():.1f}자리")
for name, dt in [("FP32", torch.float32), ("FP16", torch.float16), ("BF16", torch.bfloat16)]:
    big = torch.tensor(70000.0, dtype=dt)                                       # FP16의 최댓값(65504)보다 큰 수
    small = torch.tensor(1.0, dtype=dt) + torch.tensor(0.001, dtype=dt)        # 1에 0.001 더하기
    acc = torch.tensor(0.0, dtype=dt)
    for _ in range(10000):                                                      # 0.0001을 1만 번 더하면 1이어야 한다
        acc = acc + torch.tensor(0.0001, dtype=dt)
    print(f"{name}: 70000 → {big.item():g} | 1 + 0.001 → {small.item():.6g} | 0.0001을 1만 번 더함 → {acc.item():.4g}")

# 2. YOLO26s — 가중치와 Activation
dev = "cuda"
net = YOLO("yolo26s.pt").model.to(dev).eval()
n_params = sum(p.numel() for p in net.parameters())
print(f"\nYOLO26s 파라미터 {n_params:,}개 → FP32 {n_params * 4 / MiB:.1f} MiB, FP16 {n_params * 2 / MiB:.1f} MiB")
print(f"{'정밀도':<6}{'배치':>5}{'가중치(MiB)':>12}{'최대 메모리(MiB)':>16}{'Activation(MiB)':>16}{'1장(ms)':>9}")
for dtype in [torch.float32, torch.float16]:
    net = net.to(dtype)
    for bs in [1, 8, 32]:
        x = torch.randn(bs, 3, 640, 640, device=dev, dtype=dtype)
        with torch.inference_mode():
            net(x)                                                              # 워밍업
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()                                # 가중치 + 입력
            t = time.perf_counter()
            for _ in range(5):
                net(x)
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 5 / bs * 1000
        w = sum(p.numel() * p.element_size() for p in net.parameters())
        peak = torch.cuda.max_memory_allocated()
        print(f"{'FP32' if dtype == torch.float32 else 'FP16':<6}{bs:>5}{w / MiB:>12.1f}{peak / MiB:>16.1f}{(peak - base) / MiB:>16.1f}{ms:>9.2f}")
        del x
del net
torch.cuda.empty_cache()


# 3. Qwen3.5 — KV Cache와 선형 Attention 상태 (설정 파일로 계산)
def cache_sizes(name, state_bytes=4):                                         # 선형 상태는 FP32(4바이트)로 보관
    c = AutoConfig.from_pretrained(name).text_config
    full = c.layer_types.count("full_attention")
    linear = c.layer_types.count("linear_attention")
    kv_per_token = full * 2 * c.num_key_value_heads * c.head_dim * 2            # K와 V, BF16(2바이트)
    all_full = c.num_hidden_layers * 2 * c.num_key_value_heads * c.head_dim * 2  # 모든 층이 일반 Attention이었다면
    state = linear * c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim * state_bytes
    return full, linear, kv_per_token, all_full, state


print(f"\n{'모델':<16}{'일반/선형 층':>12}{'KV(토큰당)':>12}{'KV(16,384토큰)':>16}{'모두 일반이면':>14}{'선형 상태(요청당)':>18}")
for name in ["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]:
    full, linear, kv, all_full, state = cache_sizes(name)
    print(f"{name.split('/')[-1]:<16}{f'{full}/{linear}':>12}{f'{kv / 1024:.0f} KiB':>12}{f'{kv * 16384 / MiB:.0f} MiB':>16}"
          f"{f'{all_full * 16384 / MiB:.0f} MiB':>14}{f'{state / MiB:.0f} MiB':>18}")

# 3-2. 실제로 불러와 측정 — 가중치, 그리고 이미지를 넣은 뒤 캐시에 쌓인 메모리
proc = AutoProcessor.from_pretrained(args.vlm)
model = AutoModelForImageTextToText.from_pretrained(args.vlm, dtype=torch.bfloat16, device_map=dev).eval()
weights = torch.cuda.memory_allocated()
print(f"\n{args.vlm} BF16 가중치: {weights / GiB:.2f} GiB")
cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 1448)
frame = Image.fromarray(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
for label, img in [("bus.jpg 810x1080", Image.open(ROOT / "data" / "images" / "bus.jpg")), ("프레임 1920x1080", frame)]:
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe this image."}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                      return_tensors="pt", enable_thinking=False).to(dev)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model(**inputs, use_cache=True)                                   # 입력 전체를 한 번 읽기(prefill)
    kv = st = 0
    for layer in out.past_key_values.layers:                                    # 층마다 캐시: 일반 Attention은 K·V, 선형은 상태
        for key, v in vars(layer).items():
            if key in ("keys", "values"):
                kv += v.numel() * v.element_size()
            elif key in ("conv_states", "recurrent_states"):                    # {상태 번호: 텐서}
                st += sum(t.numel() * t.element_size() for t in v.values())
    n = inputs["input_ids"].shape[1]
    print(f"  {label}: 입력 {n} 토큰 | KV Cache {kv / MiB:.1f} MiB (토큰당 {kv / n / 1024:.0f} KiB) | 선형 상태 {st / MiB:.1f} MiB | "
          f"최대 메모리 {torch.cuda.max_memory_allocated() / GiB:.2f} GiB (가중치 + {(torch.cuda.max_memory_allocated() - weights) / GiB:.2f})")
    del out
    torch.cuda.empty_cache()
