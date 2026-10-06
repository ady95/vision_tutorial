"""16-1 프로젝트 3단계 — 세 가지 검사기를 같은 현장 평가셋으로 비교하고, 검사 리포트를 만든다

    A. Rule (02-6): 치수·구멍 수·꼭짓점 규칙 + 조명 보정
    B. CNN 분류 (03-5): 정상/불량 두 가지로 분류
    C. 결함 영역 분할 + 치수 규칙: YOLO26n-seg가 찾은 결함 영역(구멍 누락·깨짐·균열) + 02-6의 치수 측정(회전·위치·크기)
평가: 12-3의 현장 평가셋(조건 7개 x 60장) / 결함 영역의 정확도는 parts_seg 검증셋(정답 마스크가 있음)으로

실행 (저장소 루트에서, 02-6·03-5 실습, 12-3, make_data.py, train_seg.py를 마친 뒤):
    python ch16_projects/p1_defect_inspection/inspect_compare.py
"""
import contextlib
import csv
import importlib.util
import io
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ch03_deep_learning"))
from parts_cls import MEAN, STD  # noqa: E402

BENCH, SEG = ROOT / "data" / "parts_bench", ROOT / "data" / "parts_seg"
OUT = ROOT / "outputs" / "ch16" / "p1"
OUT.mkdir(parents=True, exist_ok=True)

# 검사기 준비 — 앞 장의 것을 그대로
spec = importlib.util.spec_from_file_location("rule", ROOT / "ch02_rule_based" / "02_6_rule_inspection.py")
rule = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(rule)
cnn = models.resnet18()
cnn.fc = nn.Linear(cnn.fc.in_features, 2)
cnn.load_state_dict(torch.load(ROOT / "outputs" / "ch03" / "resnet18_parts.pt", map_location="cpu"))
cnn = cnn.cuda().eval()
norm = transforms.Normalize(MEAN, STD)
seg = YOLO(ROOT / "outputs" / "ch16" / "p1" / "seg" / "weights" / "best.pt")


def check_rule(gray):
    """(불량인가, 이유 목록)"""
    fails = rule.judge(rule.measure(gray, True), rule.spec, True)
    return bool(fails), fails


def check_cnn(gray):
    x = cv2.resize(gray, (224, 168), interpolation=cv2.INTER_AREA)
    x = norm(torch.from_numpy(x).float().div(255).unsqueeze(0).repeat(3, 1, 1)).unsqueeze(0).cuda()
    with torch.no_grad():
        p = cnn(x).softmax(1)[0, 1].item()                                    # 불량일 확률
    return p >= 0.5, [f"불량 확률 {p:.2f}"]


def check_seg(gray):
    """결함 영역(분할) + 치수 규칙(회전·위치·크기). (불량인가, 이유, 결함 마스크)"""
    r = seg(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), conf=0.25, retina_masks=True, verbose=False)[0]
    reasons, mask = [], np.zeros(gray.shape, np.uint8)
    if r.masks is not None:
        for m, c, s in zip(r.masks.data.cpu().numpy(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            mask[m > 0.5] = 255
            reasons.append(f"{r.names[int(c)]} {s:.2f} ({int((m > 0.5).sum())}화소)")
    f = rule.measure(gray, True)
    if f is None:
        reasons.append("부품 없음")
    else:
        if abs(f["angle"]) > 8:
            reasons.append(f"각도 {f['angle']:.1f}도")
        if f["offset"] > 40:
            reasons.append(f"위치 {f['offset']:.0f}화소")
        if abs(f["area"] / rule.spec["area"] - 1) > 0.06 and r.masks is None:  # 결함 영역 때문이 아닌 크기 차이
            reasons.append(f"크기 {f['area'] / rule.spec['area']:.2f}배")
    return bool(reasons), reasons, mask


# 1. 현장 평가셋 — 정확도, 불량 유형별, 속도
with open(BENCH / "labels.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
images = {r["file"]: cv2.imread(str(BENCH / r["file"]), cv2.IMREAD_GRAYSCALE) for r in rows}
METHODS = {"A. Rule": check_rule, "B. CNN": check_cnn, "C. 분할 + 치수": lambda g: check_seg(g)[:2]}
results = {}
for name, fn in METHODS.items():
    fn(images[rows[0]["file"]])                                               # 워밍업
    t = time.perf_counter()
    results[name] = {r["file"]: fn(images[r["file"]])[0] for r in rows}
    results[name]["_ms"] = (time.perf_counter() - t) / len(rows) * 1000
conds = list(dict.fromkeys(r["light"] for r in rows))
print(f"현장 평가셋 {len(rows)}장 (12-3)\n{'방법':<14}" + "".join(f"{c:>9}" for c in conds) + f"{'전체':>8}{'정상 오판':>9}{'ms/장':>8}")
for name, res in results.items():
    acc = [np.mean([res[r["file"]] == (r["label"] != "ok") for r in rows if r["light"] == c]) for c in conds]
    total = np.mean([res[r["file"]] == (r["label"] != "ok") for r in rows])
    fpr = np.mean([res[r["file"]] for r in rows if r["label"] == "ok"])
    print(f"{name:<14}" + "".join(f"{a:>9.1%}" for a in acc) + f"{total:>8.1%}{fpr:>9.1%}{res['_ms']:>8.1f}")
print(f"\n{'불량 유형별 검출':<14}" + "".join(f"{d:>13}" for d in ["missing_hole", "chip", "crack", "rotated", "offset", "small"]))
for name, res in results.items():
    per = defaultdict(list)
    for r in rows:
        per[r["label"]].append(res[r["file"]])
    print(f"{name:<14}" + "".join(f"{np.mean(per[d]):>13.0%}" for d in ["missing_hole", "chip", "crack", "rotated", "offset", "small"]))

# 2. 결함 영역의 정확도 — 정답 마스크가 있는 parts_seg 검증셋
ious = defaultdict(list)
for p in sorted((SEG / "images" / "val").glob("*.png")):
    label = p.stem.split("_", 2)[2]
    if label not in ("missing_hole", "chip", "crack"):
        continue
    gt = cv2.imread(str(SEG / "masks" / "val" / p.name), cv2.IMREAD_GRAYSCALE) > 0
    pred = check_seg(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))[2] > 0
    ious[label].append((gt & pred).sum() / max(1, (gt | pred).sum()))
print("\n결함 영역 IoU (parts_seg 검증셋, 정답 마스크와 비교)")
for k, v in ious.items():
    print(f"  {k:<13} {len(v)}장: 평균 {np.mean(v):.3f}, IoU 0.5 이상 {np.mean([x >= 0.5 for x in v]):.0%}")

# 3. 검사 리포트 — 조건별로 한 장씩, 세 검사기의 판정과 이유
picks = [r for r in rows if r["label"] in ("chip", "crack", "missing_hole")][::25][:4]
tiles = []
for r in picks:
    gray = images[r["file"]]
    is_def, reasons, mask = check_seg(gray)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vis[mask > 0] = (0.4 * vis[mask > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    lines = [f"{r['light']} / 정답 {r['label']}",
             f"A Rule: {'불량' if check_rule(gray)[0] else '정상'}",
             f"B CNN: {'불량' if check_cnn(gray)[0] else '정상'}",
             f"C 분할: {'불량' if is_def else '정상'}"]
    print(f"[리포트] {r['file']}: " + " | ".join(lines[1:]) + f" | C의 이유 {reasons}")
    tiles.append(cv2.resize(vis, (320, 240)))
cv2.imwrite(str(OUT / "report.jpg"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]))
