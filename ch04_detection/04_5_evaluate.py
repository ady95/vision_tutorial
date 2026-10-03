"""04-5. Detection 모델 평가와 Threshold 조정 — 그리고 검출 결과로 정상/불량 판정하기

실행 (저장소 루트에서, 04-4 학습 후):
    python ch04_detection/04_5_evaluate.py
"""
import csv
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DET = ROOT / "data" / "parts_det"
PARTS = ROOT / "data" / "parts"
NAMES = ["part", "hole", "chip", "crack"]
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(ROOT / "outputs" / "ch04" / "runs" / "finetune" / "weights" / "best.pt")

# 1. Ultralytics가 계산한 클래스별 지표 (시험 150장)
m = model.val(data=str(DET / "data.yaml"), split="test", device=device, verbose=False, plots=False)
print(f"{'class':<7}{'P':>7}{'R':>7}{'AP50':>8}{'AP50-95':>9}")
for i, n in enumerate(NAMES):
    print(f"{n:<7}{m.box.p[i]:>7.3f}{m.box.r[i]:>7.3f}{m.box.ap50[i]:>8.3f}{m.box.ap[i]:>9.3f}")


# 2. 신뢰도 기준(conf)과 IoU 기준에 따른 TP·FP·FN — 예측과 정답을 직접 짝지어 센다
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def load_gt(label_file, w=640, h=480):
    gt = []
    for line in label_file.read_text().split("\n"):
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split())
            gt.append((int(c), [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return gt


images = sorted((DET / "images" / "test").glob("*.png"))
gts = {f: load_gt(DET / "labels" / "test" / f"{f.stem}.txt") for f in images}


def count(preds, conf, iou_th):
    """신뢰도 conf 이상인 예측을, 점수가 높은 것부터 아직 짝이 없는 같은 클래스 정답과 IoU iou_th 이상이면 짝짓는다"""
    cnt = defaultdict(Counter)
    for f in images:
        gt, r = gts[f], preds[f]
        dets = sorted([(float(s), int(c), b) for s, c, b in zip(r.boxes.conf, r.boxes.cls, r.boxes.xyxy.tolist())
                       if s >= conf], reverse=True)
        used = set()
        for s, c, b in dets:
            match = max(((iou(b, g), j) for j, (gc, g) in enumerate(gt) if gc == c and j not in used), default=(0, -1))
            if match[0] >= iou_th:
                used.add(match[1])
                cnt[c]["TP"] += 1
            else:
                cnt[c]["FP"] += 1
        for j, (gc, _) in enumerate(gt):
            if j not in used:
                cnt[gc]["FN"] += 1
    return " | ".join(f"{n} {cnt[c]['TP']}/{cnt[c]['FP']}/{cnt[c]['FN']}" for c, n in enumerate(NAMES))


scratch = YOLO(ROOT / "outputs" / "ch04" / "runs" / "scratch" / "weights" / "best.pt")
for name, mdl in [("미세조정 모델", model), ("처음부터 학습한 모델", scratch)]:
    preds = {f: mdl(f, device=device, conf=0.01, verbose=False)[0] for f in images}
    print(f"\n[{name}] conf 기준별 TP / FP / FN (IoU 0.5 이상이면 정답)")
    for conf in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"  conf {conf:<5}{count(preds, conf, 0.5)}")
    if mdl is model:
        print(f"[{name}] 정답으로 인정하는 IoU 기준별 (conf 0.25)")
        for th in [0.5, 0.75, 0.9]:
            print(f"  IoU {th:<5} {count(preds, 0.25, th)}")


# 3. 검출 결과로 정상/불량 판정 — 02-6과 같은 시험 180장
def features(r):
    boxes = [(int(c), b) for c, b in zip(r.boxes.cls, r.boxes.xyxy.tolist())]
    parts = [b for c, b in boxes if c == 0]
    if len(parts) != 1:
        return None
    x1, y1, x2, y2 = parts[0]
    inside = lambda b: x1 <= (b[0] + b[2]) / 2 <= x2 and y1 <= (b[1] + b[3]) / 2 <= y2
    return {
        "holes": sum(1 for c, b in boxes if c == 1 and inside(b)),
        "chip": sum(1 for c, b in boxes if c == 2), "crack": sum(1 for c, b in boxes if c == 3),
        "area": (x2 - x1) * (y2 - y1), "aspect": (x2 - x1) / (y2 - y1),
        "offset": math.dist(((x1 + x2) / 2, (y1 + y2) / 2), (320, 240)),
    }


with open(PARTS / "labels.csv", encoding="utf-8") as fp:
    rows = list(csv.DictReader(fp))
feats = {r["file"]: features(model(PARTS / r["file"], device=device, conf=0.25, verbose=False)[0]) for r in rows}
ok = [feats[r["file"]] for r in rows if r["light"] == "normal" and r["label"] == "ok"]
spec = {"area": float(np.median([f["area"] for f in ok])), "aspect_min": min(f["aspect"] for f in ok) * 0.95}
print(f"\n정상품(normal) 박스: 넓이 중앙값 {spec['area']:.0f}, 가로세로 비 {min(f['aspect'] for f in ok):.2f}~"
      f"{max(f['aspect'] for f in ok):.2f} → 기울어짐 기준 {spec['aspect_min']:.2f} 미만")


def judge(f):
    if f is None:
        return ["부품 수"]
    fails = []
    if f["holes"] != 2:
        fails.append("구멍 수")
    if f["chip"]:
        fails.append("깨짐")
    if f["crack"]:
        fails.append("균열")
    if abs(f["area"] / spec["area"] - 1) > 0.12:
        fails.append("크기")
    if f["aspect"] < spec["aspect_min"]:
        fails.append("기울어짐")
    if f["offset"] > 40:
        fails.append("위치")
    return fails


print(f"\n{'조명':<9}{'정확도':>7}{'불량 놓침':>9}{'정상 오판':>9}   불량 유형별 검출")
for light in ["normal", "dim", "gradient"]:
    target = [r for r in rows if r["light"] == light]
    per = defaultdict(lambda: [0, 0])
    missed = rejected = 0
    reasons = Counter()
    for r in target:
        fails = judge(feats[r["file"]])
        if r["label"] == "ok":
            rejected += bool(fails)
            reasons.update(fails)
        else:
            per[r["label"]][0] += bool(fails)
            per[r["label"]][1] += 1
            missed += not fails
    acc = 1 - (missed + rejected) / len(target)
    print(f"{light:<9}{acc:>7.1%}{missed:>9}{rejected:>9}   " + " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(per.items()))
          + (f"  오판 이유 {dict(reasons)}" if reasons else ""))

# 4. 속도 — 부품 이미지 1장 (GPU, 워밍업 후 20회 중앙값)
x = PARTS / "normal" / "000_ok.png"
times = []
for i in range(23):
    t = time.perf_counter()
    model(x, device=device, verbose=False)
    if i >= 3:
        times.append((time.perf_counter() - t) * 1000)
print(f"\n1장 검출 (전처리·후처리 포함): {sorted(times)[10]:.1f} ms")
