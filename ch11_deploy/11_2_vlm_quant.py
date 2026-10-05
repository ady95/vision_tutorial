"""11-2. VLM Quantization과 품질 비교 — 같은 질문을 BF16과 양자화 모델에 묻는다

10장의 질문을 줄여 한 번에 재는 시험 세트 (모두 thinking 끔):
1) 속도 — bus.jpg 설명 (입력·출력 토큰, 초당 토큰)
2) 개수 — COCO 사람 세기 263장 (10-3)
3) 관계 — 좌우·위아래 120문항 (10-5)
4) 없는 물체 — POPE adversarial 1,470문항 (10-7)
5) 좌표 — COCO 앞 100장의 "모두 찾아 좌표로" (10-4)

답은 outputs/ch11/vlm_answers_<tag>.json에 저장하고, --ref로 기준(BF16) 답과 얼마나 같은지도 센다.

실행 (저장소 루트에서, 서버를 띄운 뒤):
    python ch11_deploy/11_2_vlm_quant.py --tag 9b-bf16
    python ch11_deploy/11_2_vlm_quant.py --tag 9b-w4a16-3060 --base-url http://localhost:8001/v1 --ref 9b-bf16
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ch10_vlm"))
from vlm import ask, first_int, model_name  # noqa: E402

COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
OUT = ROOT / "outputs" / "ch11"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--tag", required=True, help="결과 파일 이름 (예: 9b-bf16)")
ap.add_argument("--ref", default="", help="답을 비교할 기준 tag")
args = ap.parse_args()
B = args.base_url
print(f"모델: {model_name(B)} ({args.tag})")
inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
names = {c["id"]: c["name"] for c in inst["categories"]}
anns = defaultdict(list)
for a in inst["annotations"]:
    anns[a["image_id"]].append(a)
image = {im["id"]: im for im in inst["images"]}
load = lambda im: Image.open(COCO_DIR / "images" / im["file_name"])  # noqa: E731
answers = {}                                                                  # 질문 id → 답 (기준 모델과 비교용)

# 1. 속도
bus = Image.open(ROOT / "data" / "images" / "bus.jpg")
ask("Describe this image.", [bus], max_tokens=16, base_url=B)                # 워밍업
r = ask("Describe this image in detail.", [bus], max_tokens=256, base_url=B)
print(f"[속도] bus.jpg: {r['seconds']:.2f}초 | 입력 {r['prompt_tokens']} / 출력 {r['completion_tokens']} 토큰 | "
      f"초당 {r['completion_tokens'] / r['seconds']:.1f} 토큰")

# 2. 개수 — 사람 (crowd 표시가 있는 사진은 뺀다)
t, res = time.perf_counter(), []
for im in inst["images"]:
    a = anns[im["id"]]
    gt = sum(x["category_id"] == 1 and not x["iscrowd"] for x in a)
    if gt == 0 or any(x["category_id"] == 1 and x["iscrowd"] for x in a):
        continue
    ans = first_int(ask("How many people are in this image? Answer with a number only.", [load(im)], max_tokens=16, base_url=B)["text"])
    answers[f"count/{im['id']}"] = ans
    res.append((gt, ans))
print(f"[개수] 사람 {len(res)}장: 정확히 맞힘 {np.mean([g == a for g, a in res]):.1%}, "
      f"평균 오차 {np.mean([abs((a or 0) - g) for g, a in res]):.2f} ({time.perf_counter() - t:.0f}초)")

# 3. 관계 — 10-5와 같은 120문항
rng = random.Random(0)
qs = []
for im in inst["images"]:
    single = [a for a in anns[im["id"]] if not a["iscrowd"] and a["area"] > 32 ** 2
              and sum(b["category_id"] == a["category_id"] for b in anns[im["id"]]) == 1]
    for a, b in combinations(single, 2):
        ca = (a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2)
        cb = (b["bbox"][0] + b["bbox"][2] / 2, b["bbox"][1] + b["bbox"][3] / 2)
        na, nb = names[a["category_id"]], names[b["category_id"]]
        if abs(ca[0] - cb[0]) > 0.15 * im["width"]:
            qs.append((im, f"Is the {na} to the left of the {nb} in this image? Answer yes or no.", "yes" if ca[0] < cb[0] else "no"))
        if abs(ca[1] - cb[1]) > 0.15 * im["height"]:
            qs.append((im, f"Is the {na} above the {nb} in this image? Answer yes or no.", "yes" if ca[1] < cb[1] else "no"))
rng.shuffle(qs)
qs = [q for q in qs if q[2] == "yes"][:60] + [q for q in qs if q[2] == "no"][:60]
t, ok = time.perf_counter(), []
for i, (im, q, truth) in enumerate(qs):
    pred = "yes" if ask(q, [load(im)], max_tokens=8, base_url=B)["text"].lower().startswith("yes") else "no"
    answers[f"rel/{i}"] = pred
    ok.append(pred == truth)
print(f"[관계] {len(qs)}문항: 정확도 {np.mean(ok):.1%} ({time.perf_counter() - t:.0f}초)")

# 4. 없는 물체 — POPE adversarial (10-7과 같은 질문)
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
t, tp, fp, tn, fn = time.perf_counter(), 0, 0, 0, 0
for im, cls, truth in pope:
    pred = "yes" if ask(f"Is there a {cls} in the image? Answer yes or no.", [load(im)], max_tokens=8, base_url=B)["text"].lower().startswith("yes") else "no"
    answers[f"pope/{im['id']}/{cls}"] = pred
    tp, fp = tp + (pred == truth == "yes"), fp + (pred == "yes" != truth)
    tn, fn = tn + (pred == truth == "no"), fn + (pred == "no" != truth)
print(f"[없는 물체] POPE adversarial {len(pope)}문항: 정확도 {(tp + tn) / len(pope):.1%}, 있는 물체 재현율 {tp / (tp + fn):.1%}, "
      f"없는데 yes {fp / (fp + tn):.1%} ({time.perf_counter() - t:.0f}초)")


# 5. 좌표 — COCO 앞 100장 (10-4와 같은 질문·채점)
def parse_boxes(text, size):
    w, h = size
    out = []
    for obj in re.findall(r"\{[^{}]*\}", text):
        nums = re.search(r'"bbox_2d"\s*:\s*\[([^\]]*)\]', obj)
        v = [float(x) for x in re.findall(r"-?\d+\.?\d*", nums.group(1))] if nums else []
        if len(v) == 4:
            out.append([v[0] / 1000 * w, v[1] / 1000 * h, v[2] / 1000 * w, v[3] / 1000 * h])
    return out


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy)


def match(preds, gts):
    used_p, used_g = set(), set()
    for v, i, j in sorted(((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True):
        if v >= 0.5 and i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    return len(used_g)


rng = random.Random(0)
s, t = defaultdict(int), time.perf_counter()
for im in inst["images"][:100]:
    gt = defaultdict(list)
    for a in anns[im["id"]]:
        if not a["iscrowd"]:
            x, y, w, h = a["bbox"]
            gt[names[a["category_id"]]].append([x, y, x + w, y + h])
    absent = rng.choice([n for n in names.values() if n not in gt])
    for cls in list(gt) + [absent]:
        text = ask(f'Find every {cls} in this image. Output a JSON list like [{{"label": "{cls}", "bbox_2d": [x1, y1, x2, y2]}}]. '
                   f"If there is no {cls}, output [].", [load(im)], max_tokens=1024, base_url=B)["text"]
        boxes = parse_boxes(text, (im["width"], im["height"]))
        answers[f"box/{im['id']}/{cls}"] = len(boxes)
        if cls == absent:
            s["없음"] += 1
            s["지어냄"] += int(len(boxes) > 0)
        else:
            s["정답"] += len(gt[cls])
            s["예측"] += len(boxes)
            s["맞힘"] += match(boxes, gt[cls])
print(f"[좌표] COCO 100장 {sum(k.startswith('box/') for k in answers)}문항: 재현율 {s['맞힘'] / s['정답']:.1%}, "
      f"정밀도 {s['맞힘'] / max(1, s['예측']):.1%}, 없는 물체를 찾아냄 {s['지어냄'] / s['없음']:.1%} ({time.perf_counter() - t:.0f}초)")

# 저장과 기준 모델 비교
(OUT / f"vlm_answers_{args.tag}.json").write_text(json.dumps(answers), encoding="utf-8")
ref_path = OUT / f"vlm_answers_{args.ref}.json"
if args.ref and ref_path.exists():
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    for kind in ["count", "rel", "pope", "box"]:
        keys = [k for k in answers if k.startswith(kind + "/") and k in ref]
        print(f"  {args.ref}와 같은 답 [{kind}]: {np.mean([answers[k] == ref[k] for k in keys]):.1%} ({len(keys)}문항)")
