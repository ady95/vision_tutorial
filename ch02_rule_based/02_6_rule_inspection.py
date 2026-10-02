"""02-6. 실습: Rule 기반 제품 검사 시스템

Grayscale → Threshold → Morphology → Contour → 크기·위치·각도·형태 계산 → Pass/Fail

기준값은 "정상 시료"(normal 조명의 정상품)에서 정하고, 조명이 바뀐 이미지에 그대로 적용해 본다.

실행:
    python data/make_parts.py
    python ch02_rule_based/02_6_rule_inspection.py
"""
import csv
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "data" / "parts"
OUT = ROOT / "outputs" / "ch02"
OUT.mkdir(parents=True, exist_ok=True)
CENTER = (320, 240)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
BG_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (171, 171))


def binarize(gray, correct_light=False):
    if correct_light:                                     # 02-2의 조명 보정
        bg = cv2.morphologyEx(cv2.GaussianBlur(gray, (9, 9), 0), cv2.MORPH_OPEN, BG_KERNEL)
        bg = cv2.GaussianBlur(bg, (0, 0), 25)
        gray = cv2.divide(gray, cv2.max(bg, 1), scale=64)
    _, b = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(b, cv2.MORPH_OPEN, KERNEL)


def measure(gray, correct_light=False):
    """부품 하나의 특징값. 부품을 찾지 못하면 None"""
    b = binarize(gray, correct_light)
    contours, hierarchy = cv2.findContours(b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    hierarchy = hierarchy[0]
    outers = [i for i in range(len(contours)) if hierarchy[i][3] == -1]
    outer = max(outers, key=lambda i: cv2.contourArea(contours[i]))
    c = contours[outer]
    area = cv2.contourArea(c)
    M = cv2.moments(c)
    holes = [i for i in range(len(contours)) if hierarchy[i][3] == outer and cv2.contourArea(contours[i]) > 200]
    return {
        "area": area,
        "holes": len(holes),
        "solidity": area / cv2.contourArea(cv2.convexHull(c)),
        "compactness": cv2.arcLength(c, True) / math.sqrt(area),   # 둘레가 면적에 비해 얼마나 긴가
        "angle": 0.5 * math.degrees(math.atan2(2 * M["mu11"], M["mu20"] - M["mu02"])),
        "offset": math.dist((M["m10"] / M["m00"], M["m01"] / M["m00"]), CENTER),
        "corners": len(cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)),
    }


def judge(f, spec, corner_rule=False):
    """규칙을 하나씩 확인하고 어긴 규칙 이름을 돌려준다 (빈 목록이면 Pass)"""
    if f is None:
        return ["부품 없음"]
    fails = []
    if abs(f["area"] / spec["area"] - 1) > 0.06:
        fails.append("크기")
    if f["holes"] != 2:
        fails.append("구멍 수")
    if f["solidity"] < spec["solidity_min"]:
        fails.append("깨짐")
    if f["compactness"] > spec["compactness_max"]:
        fails.append("균열")
    if abs(f["angle"]) > 8:
        fails.append("각도")
    if f["offset"] > 40:
        fails.append("위치")
    if corner_rule and f["corners"] != 4:          # 3단계에서 추가하는 규칙
        fails.append("꼭짓점")
    return fails


with open(PARTS / "labels.csv", encoding="utf-8") as fp:
    rows = list(csv.DictReader(fp))

# 1. 기준값 정하기 — normal 조명의 정상품에서 특징값 분포를 보고 여유를 둔다
ok_feats = [measure(cv2.imread(str(PARTS / r["file"]), cv2.IMREAD_GRAYSCALE))
            for r in rows if r["light"] == "normal" and r["label"] == "ok"]
spec = {
    "area": float(np.median([f["area"] for f in ok_feats])),
    "solidity_min": min(f["solidity"] for f in ok_feats) - 0.01,
    "compactness_max": max(f["compactness"] for f in ok_feats) * 1.03,
}
print("정상품 30장의 특징값 범위")
for key in ["area", "solidity", "compactness", "angle", "offset"]:
    v = [f[key] for f in ok_feats]
    print(f"  {key:<12} {min(v):9.3f} ~ {max(v):9.3f}")
print("정한 기준:", {k: round(v, 3) for k, v in spec.items()}, "+ 크기 ±6%, 구멍 2개, 각도 ±8도, 위치 40px 이내")


# 2. 검사 — 조명별, 조명 보정 유무별
def inspect(light, correct_light, corner_rule=False):
    stats, per_defect, reasons = Counter(), defaultdict(lambda: [0, 0]), Counter()
    t = time.perf_counter()
    targets = [r for r in rows if r["light"] == light]
    for r in targets:
        gray = cv2.imread(str(PARTS / r["file"]), cv2.IMREAD_GRAYSCALE)
        fails = judge(measure(gray, correct_light), spec, corner_rule)
        is_defect, caught = r["label"] != "ok", bool(fails)
        stats[("불량" if is_defect else "정상", "Fail" if caught else "Pass")] += 1
        if is_defect:
            per_defect[r["label"]][0] += caught
            per_defect[r["label"]][1] += 1
        elif caught:
            reasons.update(fails)
    ms = (time.perf_counter() - t) * 1000 / len(targets)
    acc = (stats[("정상", "Pass")] + stats[("불량", "Fail")]) / len(targets)
    return acc, stats, per_defect, reasons, ms


def report(title, cases):
    print(f"\n{title}")
    print(f"{'조명':<9}{'조명 보정':<9}{'정확도':>7}{'불량 놓침':>9}{'정상 오판':>9}{'ms/장':>8}   불량 유형별 검출")
    for light, correct, corner_rule in cases:
        acc, stats, per_defect, reasons, ms = inspect(light, correct, corner_rule)
        caught = " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(per_defect.items()))
        print(f"{light:<9}{'O' if correct else '-':<9}{acc:>7.1%}{stats[('불량', 'Pass')]:>9}"
              f"{stats[('정상', 'Fail')]:>9}{ms:>8.1f}   {caught}")
        if reasons:
            print(f"{'':<18}정상품을 불량으로 본 이유: {dict(reasons)}")


report("[1차 규칙] 조명이 바뀌면?", [(light, False, False) for light in ["normal", "dim", "gradient"]])
report("[1차 규칙 + 조명 보정]", [(light, True, False) for light in ["normal", "dim", "gradient"]])

# 3. 놓친 불량을 보고 규칙 추가 — 모서리를 곧게 잘라 낸 깨짐(chip)은 여전히 볼록해서 Solidity로 못 잡는다
chip = next(r for r in rows if r["light"] == "normal" and r["label"] == "chip")
ok = next(r for r in rows if r["light"] == "normal" and r["label"] == "ok")
for r in [ok, chip]:
    f = measure(cv2.imread(str(PARTS / r["file"]), cv2.IMREAD_GRAYSCALE))
    print(f"\n{r['label']:<5}: solidity {f['solidity']:.3f} | 면적비 {f['area'] / spec['area']:.3f} | 꼭짓점 {f['corners']}개", end="")
print()
report("[2차 규칙: 꼭짓점 수 추가 + 조명 보정]", [(light, True, True) for light in ["normal", "dim", "gradient"]])

# 4. 실패 사례 저장 — gradient 정상품의 이진화 결과 (원본 | 보정 없음 | 보정)
r = next(r for r in rows if r["light"] == "gradient" and r["label"] == "ok")
gray = cv2.imread(str(PARTS / r["file"]), cv2.IMREAD_GRAYSCALE)
cv2.imwrite(str(OUT / "02_6_gradient_ok.png"), np.hstack([gray, binarize(gray), binarize(gray, True)]))
