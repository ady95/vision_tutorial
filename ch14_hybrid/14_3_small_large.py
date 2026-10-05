"""14-3. Small Model + Large Model — Qwen3.5-2B로 거르고, 애매한 것만 9B에게

질문: 10-7·11-2와 같은 POPE adversarial 1,470문항 ("Is there a X in the image?")
1) 2B는 답의 첫 토큰 확률(logprobs)에서 yes와 no의 확률을 꺼내 "확신도"로 쓴다
2) 확신도가 기준보다 낮은 질문만 9B에게 다시 묻는다 — 기준을 바꿔 가며 정확도와 9B 호출 비율, 시간
3) 고른 기준으로 실제로 두 서버를 차례로 불러 확인

실행 (저장소 루트에서, 2B와 9B 서버를 띄운 뒤):
    python ch14_hybrid/14_3_small_large.py --small-url http://localhost:8001/v1 --large-url http://localhost:8000/v1
"""
import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ch10_vlm"))
from vlm import client, encode, model_name  # noqa: E402

COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
OUT = ROOT / "outputs" / "ch14"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--small-url", default="http://localhost:8001/v1")
ap.add_argument("--large-url", default="http://localhost:8000/v1")
args = ap.parse_args()

# 질문 — 11-2와 같은 POPE adversarial
inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
names = {c["id"]: c["name"] for c in inst["categories"]}
present = defaultdict(set)
for a in inst["annotations"]:
    present[a["image_id"]].add(names[a["category_id"]])
co = Counter()
for s in present.values():
    for a, b in combinations(sorted(s), 2):
        co[(a, b)] += 1
        co[(b, a)] += 1
rng = random.Random(0)
pope = []
for im in [im for im in inst["images"] if len(present[im["id"]]) >= 3]:
    have = present[im["id"]]
    pos = rng.sample(sorted(have), 3)
    missing = [n for n in names.values() if n not in have]
    rng.sample(missing, 3)                                                    # 10-7의 random 방식과 난수 순서를 맞춘다
    pope += [(im, c, "yes") for c in pos] + [(im, c, "no") for c in sorted(missing, key=lambda n: -sum(co[(h, n)] for h in have))[:3]]


def ask_yes(url, im, cls):
    """yes/no 한 토큰만 받고, 첫 토큰 후보의 확률에서 P(yes)를 계산한다"""
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": encode(Image.open(COCO_DIR / "images" / im["file_name"]))}},
                                         {"type": "text", "text": f"Is there a {cls} in the image? Answer yes or no."}]}]
    r = client(url).chat.completions.create(model=model_name(url), messages=msgs, max_tokens=1, temperature=0.0,
                                            logprobs=True, top_logprobs=10,
                                            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    p = defaultdict(float)
    for t in r.choices[0].logprobs.content[0].top_logprobs:
        word = t.token.strip().lower()
        if word in ("yes", "no"):
            p[word] += math.exp(t.logprob)
    return p["yes"] / max(1e-9, p["yes"] + p["no"])


# 1. 두 모델의 답을 한 번씩 (나중의 Routing 계산에 쓴다)
cache = OUT / "14_3_answers.json"
if cache.exists():
    C = json.loads(cache.read_text(encoding="utf-8"))
else:
    C = {"ms": {}, "p": {}}
    for key, url in [("small", args.small_url), ("large", args.large_url)]:
        ask_yes(url, pope[0][0], pope[0][1])                                  # 워밍업
        t = time.perf_counter()
        C["p"][key] = [ask_yes(url, im, cls) for im, cls, _ in pope]
        C["ms"][key] = (time.perf_counter() - t) / len(pope) * 1000
    cache.write_text(json.dumps(C), encoding="utf-8")
truth = np.array([t == "yes" for _, _, t in pope])
ps, pl = np.array(C["p"]["small"]), np.array(C["p"]["large"])
ms = C["ms"]
print(f"POPE adversarial {len(pope)}문항 | 질문당 시간: 2B {ms['small']:.0f}ms, 9B {ms['large']:.0f}ms")


def report(label, pred, frac_large, ms_per_q):
    """정확도, 없는데 yes라고 한 비율, 9B를 부른 비율, 질문당 평균 시간"""
    print(f"{label:<30}{np.mean(pred == truth):>8.1%}{np.mean(pred[~truth]):>11.1%}{frac_large:>10.1%}{ms_per_q:>10.0f}")


# 2. Routing 기준 — 2B의 확신도 max(P(yes), P(no))가 기준보다 낮으면 9B에게
print(f"\n{'구성':<30}{'정확도':>8}{'없는데 yes':>11}{'9B 호출':>10}{'ms/질문':>10}")
report("2B 단독", ps >= 0.5, 0.0, ms["small"])
report("9B 단독", pl >= 0.5, 1.0, ms["large"])
conf = np.maximum(ps, 1 - ps)
for th in [0.6, 0.8, 0.9, 0.95, 0.99, 0.999]:
    route = conf < th
    report(f"2B → 확신도 {th} 미만만 9B", np.where(route, pl >= 0.5, ps >= 0.5), route.mean(), ms["small"] + route.mean() * ms["large"])
print("\n2B 확신도 구간별 2B의 정확도 (확신도가 정말 정답률을 뜻하는가)")
for lo, hi in [(0.5, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1.01)]:
    sel = (conf >= lo) & (conf < hi)
    print(f"  {lo}~{min(hi, 1)}: {sel.sum():>5}문항, 2B 정확도 {np.mean((ps >= 0.5)[sel] == truth[sel]):.1%}, "
          f"9B 정확도 {np.mean((pl >= 0.5)[sel] == truth[sel]):.1%}")

# 3. 실제로 차례로 불러 보기 (기준 0.99)
t, calls, correct = time.perf_counter(), 0, 0
for (im, cls, ans) in pope:
    p = ask_yes(args.small_url, im, cls)
    if max(p, 1 - p) < 0.99:
        calls += 1
        p = ask_yes(args.large_url, im, cls)
    correct += (p >= 0.5) == (ans == "yes")
sec = time.perf_counter() - t
print(f"\n실제 실행 (기준 0.99): {len(pope)}문항 {sec:.0f}초 ({sec / len(pope) * 1000:.0f}ms/질문), 9B 호출 {calls}번 "
      f"({calls / len(pope):.1%}), 정확도 {correct / len(pope):.1%}")
