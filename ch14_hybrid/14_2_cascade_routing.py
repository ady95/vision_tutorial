"""14-2. Cascade 구조와 Confidence 기반 Routing — Rule → YOLO → VLM

평가셋: 12-3의 부품 평가셋(조건 7개 x 60장) + "현장에서 처음 나타난 불량" 세트(정상 30, 얼룩 30)
1) 단계마다 판정과 "확실한가"를 한 번씩 계산해 저장한다
   - Rule(02-6): 판정 기준까지 남은 여유(margin). 구멍 수·꼭짓점 수처럼 셀 수 있는 규칙을 어기면 확실한 불량
   - YOLO(04-5): 신뢰도 0.1~0.5 사이의 애매한 박스가 있으면 확실하지 않음
   - VLM(Qwen3.5-9B): 마지막 단계
2) Routing 기준값을 바꿔 가며: 정확도, 다음 단계로 넘어간 비율, 장당 평균 시간
3) 감사(audit): "확실한 정상"의 일부를 무작위로 VLM에 보내면 처음 보는 불량을 알아챌 수 있는가

실행 (저장소 루트에서, 12-3 실행 후 VLM 서버를 띄운 뒤):
    python ch14_hybrid/14_2_cascade_routing.py --vlm-url http://localhost:8000/v1
"""
import argparse
import contextlib
import csv
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "ch10_vlm"))
import make_parts  # noqa: E402
from vlm import ask  # noqa: E402

BENCH, STAIN = ROOT / "data" / "parts_bench", ROOT / "data" / "parts_stain"
OUT = ROOT / "outputs" / "ch14"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--vlm-url", default="http://localhost:8000/v1")
args = ap.parse_args()

# 0. 평가셋 — 12-3 평가셋 + 얼룩 세트(표면에 옅은 회색 얼룩. 모양은 그대로라 치수 규칙으로는 안 보인다)
if not (STAIN / "labels.csv").exists():
    STAIN.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2032)
    rows = []
    for i in range(60):
        img, mask, _ = make_parts.make_image(rng, "ok", "normal")
        label = "ok" if i < 30 else "stain"
        if label == "stain":
            ys, xs = np.where(mask > 0)
            k = rng.integers(len(xs))
            spot = np.zeros(mask.shape, np.float32)
            cv2.ellipse(spot, (int(xs[k]), int(ys[k])), (int(rng.uniform(18, 30)), int(rng.uniform(10, 18))),
                        float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
            spot = cv2.GaussianBlur(spot, (0, 0), 3) * (mask > 0)
            img = np.clip(img * (1 - spot * 0.3), 0, 255).astype(np.uint8)          # 얼룩 부분을 30% 어둡게
        cv2.imwrite(str(STAIN / f"{i:03d}_{label}.png"), img)
        rows.append({"light": "new_stain", "file": f"{i:03d}_{label}.png", "label": label})
    with open(STAIN / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["light", "file", "label"])
        w.writeheader()
        w.writerows(rows)
items = []                                                                    # (경로, 조건, 라벨)
for folder in [BENCH, STAIN]:
    with open(folder / "labels.csv", encoding="utf-8") as f:
        items += [(folder / r["file"], r["light"], r["label"]) for r in csv.DictReader(f)]
print(f"평가셋 {len(items)}장 (12-3의 420장 + 얼룩 세트 60장), 불량 {sum(l != 'ok' for _, _, l in items)}장")


def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


# 1. 단계마다 판정 + 확실한 정도
rule = load_script(ROOT / "ch02_rule_based" / "02_6_rule_inspection.py")


def rule_stage(path):
    """(불량인가, 여유). 여유 = 연속값 규칙들 중 기준에 가장 가까운 것까지의 상대 거리 (0이면 기준선 위)"""
    f = rule.measure(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), True)
    fails = rule.judge(f, rule.spec, True)
    if f is None or f["holes"] != 2 or f["corners"] != 4:
        return bool(fails), 1.0                                               # 셀 수 있는 규칙: 어기면(또는 못 찾으면) 확실
    s = rule.spec
    margin = min(abs(0.06 - abs(f["area"] / s["area"] - 1)) / 0.06,           # 크기 ±6%
                 abs(8 - abs(f["angle"])) / 8,                                # 각도 ±8도
                 abs(40 - f["offset"]) / 40,                                  # 위치 40화소
                 abs(f["solidity"] - s["solidity_min"]) / 0.02,
                 abs(s["compactness_max"] - f["compactness"]) / (0.03 * s["compactness_max"]))
    return bool(fails), margin


yolo = load_script(ROOT / "ch04_detection" / "04_5_evaluate.py")


def yolo_stage(path):
    """(불량인가, 애매한 박스 수). 판정은 04-5와 같이 conf 0.25 이상의 박스로"""
    r = yolo.model(path, device=yolo.device, conf=0.1, verbose=False)[0]
    unsure = int(((r.boxes.conf >= 0.1) & (r.boxes.conf < 0.5)).sum())
    return bool(yolo.judge(yolo.features(r[r.boxes.conf >= 0.25]))), unsure


Q = ("This is a photo of a metal bracket on a conveyor belt. A good bracket is a rounded rectangle with exactly two round holes, "
     "no broken corners, no cracks, placed straight in the center. Is this bracket defective? Answer 'ok' or 'defect' only.")


def vlm_stage(path):
    return ask(Q, [Image.open(path)], max_tokens=8, base_url=args.vlm_url)["text"].strip().lower().startswith("defect")


cache = OUT / "14_2_stages.json"
if cache.exists():
    S = json.loads(cache.read_text(encoding="utf-8"))
else:
    S = {"ms": {}, "res": {str(p): {} for p, _, _ in items}}
    for name, fn in [("rule", rule_stage), ("yolo", yolo_stage), ("vlm", vlm_stage)]:
        fn(items[0][0])                                                       # 워밍업
        t = time.perf_counter()
        for p, _, _ in items:
            S["res"][str(p)][name] = fn(p)
        S["ms"][name] = (time.perf_counter() - t) / len(items) * 1000
    cache.write_text(json.dumps(S), encoding="utf-8")
ms = S["ms"]
print("단계별 장당 시간: " + " | ".join(f"{k} {v:.1f}ms" for k, v in ms.items()))


# 2. Routing — 기준값에 따라 어디서 끝나는가
def cascade(res, rule_margin, use_yolo=True, use_vlm=True, audit=0.0, rng=None):
    """한 장의 최종 판정과 거쳐 간 단계"""
    is_def, margin = res["rule"]
    if margin >= rule_margin or not use_yolo:
        if not is_def and audit and rng.random() < audit:                    # 확실한 정상 중 일부를 감사
            return res["vlm"], ["rule", "vlm"]
        return is_def, ["rule"]
    y_def, unsure = res["yolo"]
    if (unsure == 0 and y_def == is_def) or not use_vlm:                     # YOLO가 확실하고 Rule과 같은 답이면 끝
        return y_def, ["rule", "yolo"]
    return res["vlm"], ["rule", "yolo", "vlm"]


def evaluate(label, **kw):
    rng = np.random.default_rng(0)
    out = [(cascade(S["res"][str(p)], rng=rng, **kw), lab, cond) for p, cond, lab in items]
    acc = np.mean([d == (lab != "ok") for (d, _), lab, _ in out])
    stain = np.mean([d for (d, _), lab, _ in out if lab == "stain"])
    old = np.mean([d == (lab != "ok") for (d, _), lab, cond in out if cond != "new_stain"])
    p_y = np.mean(["yolo" in st for (_, st), _, _ in out])
    p_v = np.mean(["vlm" in st for (_, st), _, _ in out])
    cost = ms["rule"] + p_y * ms["yolo"] + p_v * ms["vlm"]
    print(f"{label:<34}{acc:>7.1%}{old:>9.1%}{stain:>8.0%}{p_y:>9.1%}{p_v:>9.1%}{cost:>9.1f}")


print(f"\n{'구성':<34}{'정확도':>7}{'기존 420':>9}{'얼룩':>8}{'YOLO로':>9}{'VLM으로':>9}{'ms/장':>9}")
for name in ["rule", "yolo", "vlm"]:
    acc = np.mean([(S["res"][str(p)][name][0] if name != "vlm" else S["res"][str(p)][name]) == (lab != "ok") for p, _, lab in items])
    stain = np.mean([(S["res"][str(p)][name][0] if name != "vlm" else S["res"][str(p)][name]) for p, _, lab in items if lab == "stain"])
    old = np.mean([(S["res"][str(p)][name][0] if name != "vlm" else S["res"][str(p)][name]) == (lab != "ok")
                   for p, cond, lab in items if cond != "new_stain"])
    print(f"{name + ' 단독':<34}{acc:>7.1%}{old:>9.1%}{stain:>8.0%}{'':>9}{'':>9}{ms[name]:>9.1f}")
for m in [0.05, 0.1, 0.2, 0.3, 0.5]:
    evaluate(f"Rule→YOLO→VLM (Rule 여유 {m})", rule_margin=m)
evaluate("Rule→YOLO (VLM 없이, 여유 0.2)", rule_margin=0.2, use_vlm=False)
for a in [0.05, 0.1, 0.2]:
    evaluate(f"여유 0.2 + 확실한 정상의 {a:.0%} 감사", rule_margin=0.2, audit=a)

# 3. 실제로 단계를 차례로 불러 보기 (여유 0.2) — 저장된 판정과 같은지, 실제 시간은?
t, calls, same = time.perf_counter(), {"yolo": 0, "vlm": 0}, 0
for p, _, _ in items:
    is_def, margin = rule_stage(p)
    final = is_def
    if margin < 0.2:
        calls["yolo"] += 1
        y_def, unsure = yolo_stage(p)
        final = y_def
        if not (unsure == 0 and y_def == is_def):
            calls["vlm"] += 1
            final = vlm_stage(p)
    same += final == cascade(S["res"][str(p)], rule_margin=0.2)[0]
sec = time.perf_counter() - t
print(f"\n실제 실행 (여유 0.2): {len(items)}장 {sec:.1f}초 ({sec / len(items) * 1000:.1f}ms/장), "
      f"YOLO {calls['yolo']}번, VLM {calls['vlm']}번 호출, 저장된 판정과 같은 것 {same}/{len(items)}")
