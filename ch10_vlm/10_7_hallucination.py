"""10-7. VLM의 Hallucination과 한계

1) 유도 질문 — 사진에 없는 것을 있는 것처럼 물으면
2) POPE 방식 채점 — "사진에 X가 있나? (yes/no)"를 COCO val2017 사진에 묻는다
   사진마다 있는 클래스 3개 + 없는 클래스 3개, 없는 클래스는 세 가지 방식으로 고른다
   - 무작위(random): 아무 클래스나
   - 흔한 것(popular): 500장 전체에서 자주 나오는 클래스 순서로
   - 함께 나오는 것(adversarial): 이 사진에 있는 물체와 자주 함께 나오는 클래스 순서로

실행 (저장소 루트에서, vLLM 서버를 띄운 뒤):
    python ch10_vlm/10_7_hallucination.py
"""
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm import ask, model_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
args = ap.parse_args()
B = args.base_url
print(f"모델: {model_name(B)}")

# 1. 유도 질문 — bus.jpg에는 기린도 개도 없다
bus = Image.open(ROOT / "data" / "images" / "bus.jpg")
for prompt in ["What color is the giraffe in this image?", "How many dogs are in this image? Answer with a number only.",
               "Is there a giraffe in this image? Answer yes or no.", "What is written on the sign held by the woman in the red dress?"]:
    r = ask(prompt, [bus], max_tokens=100, base_url=B)
    print(f"[bus.jpg] {prompt}\n  → {r['text']}")

# 2. POPE 방식 채점
inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
names = {c["id"]: c["name"] for c in inst["categories"]}
present = defaultdict(set)
for a in inst["annotations"]:
    present[a["image_id"]].add(names[a["category_id"]])                       # crowd 표시도 "있다"로 본다
freq = Counter(c for s in present.values() for c in s)
co = Counter()
for s in present.values():
    for a, b in combinations(sorted(s), 2):
        co[(a, b)] += 1
        co[(b, a)] += 1
rng = random.Random(0)
images = [im for im in inst["images"] if len(present[im["id"]]) >= 3]       # POPE처럼 물체 종류가 3개 이상인 사진만
questions = defaultdict(list)                                                 # 방식 → [(이미지, 클래스, 정답)]
for im in images:
    have = present[im["id"]]
    pos = rng.sample(sorted(have), 3)
    missing = [n for n in names.values() if n not in have]
    pools = {"random": rng.sample(missing, 3),
             "popular": sorted(missing, key=lambda n: -freq[n])[:3],
             "adversarial": sorted(missing, key=lambda n: -sum(co[(h, n)] for h in have))[:3]}
    for setting, neg in pools.items():
        questions[setting] += [(im, c, "yes") for c in pos] + [(im, c, "no") for c in neg]
print(f"\nPOPE 방식: 사진 {len(images)}장, 방식마다 질문 {len(questions['random'])}개 (있음 절반, 없음 절반)")
answers = {}                                                                  # (이미지, 클래스) → 답 (같은 질문은 한 번만)
t0 = time.perf_counter()
print(f"{'방식':<12}{'정확도':>8}{'정밀도':>8}{'재현율':>8}{'F1':>7}{'yes 비율':>9}{'없는데 yes':>11}")
for setting, qs in questions.items():
    tp = fp = tn = fn = 0
    for im, cls, truth in qs:
        key = (im["id"], cls)
        if key not in answers:
            text = ask(f"Is there a {cls} in the image? Answer yes or no.", [Image.open(COCO_DIR / "images" / im["file_name"])],
                       max_tokens=8, base_url=B)["text"].lower()
            answers[key] = "yes" if text.startswith("yes") else "no"
        pred = answers[key]
        tp += pred == "yes" and truth == "yes"
        fp += pred == "yes" and truth == "no"
        tn += pred == "no" and truth == "no"
        fn += pred == "no" and truth == "yes"
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    print(f"{setting:<12}{(tp + tn) / len(qs):>8.1%}{p:>8.1%}{r:>8.1%}{2 * p * r / max(1e-9, p + r):>7.1%}"
          f"{(tp + fp) / len(qs):>9.1%}{fp / max(1, fp + tn):>11.1%}")
print(f"(서로 다른 질문 {len(answers)}개, {time.perf_counter() - t0:.0f}초)")
wrong = Counter(c for (iid, c), a in answers.items() if a == "yes" and c not in present[iid])
print("가장 자주 지어낸 물체: " + ", ".join(f"{c} {n}회" for c, n in wrong.most_common(6)))
