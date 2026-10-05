"""12-3. 실습: 운영 데이터로 Vision Benchmark 만들기 — 2~4장, 10장의 부품 검사기를 같은 평가셋으로

1) 평가셋 만들기: 개발에 쓰지 않은 새 이미지(seed 2031) + 현장에서 생기는 변화(초점 흐림, 노이즈, JPEG 압축, 과노출)
2) 같은 평가셋으로 비교: 02-6 Rule, 03-5 CNN, 04-5 YOLO 판정, Qwen3.5 VLM(학습 없이 질문만)
3) 조건별·불량 유형별 결과, 그리고 "불량이 1%뿐인 현장"이라면 Precision은?

각 방법은 앞 장의 스크립트를 그대로 불러와(기준값·판정 함수 포함) 쓴다. 결과는 outputs/ch12/에 저장해 두고 다시 쓴다.

실행 (저장소 루트에서, 02~04장 실습을 마친 뒤):
    python ch12_evaluation/12_3_benchmark.py --methods rule cnn yolo
    python ch12_evaluation/12_3_benchmark.py --methods vlm --vlm-url http://localhost:8000/v1
    python ch12_evaluation/12_3_benchmark.py            # 저장된 결과로 표만
"""
import argparse
import contextlib
import csv
import importlib.util
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "ch10_vlm"))
import make_parts  # noqa: E402

BENCH = ROOT / "data" / "parts_bench"
OUT = ROOT / "outputs" / "ch12"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--methods", nargs="*", default=[], help="rule cnn yolo vlm 중 새로 돌릴 것")
ap.add_argument("--vlm-url", default="http://localhost:8000/v1")
args = ap.parse_args()

# 1. 평가셋 — 조건마다 60장(정상 30, 불량 6종 x 5)
CONDITIONS = {
    "normal": ("normal", None), "gradient": ("gradient", None), "dim": ("dim", None),            # 개발 때 본 조명 (새 이미지)
    "blur": ("normal", lambda im: cv2.GaussianBlur(im, (0, 0), 2.0)),                           # 초점이 조금 나감
    "noise": ("normal", lambda im, r=np.random.default_rng(1): np.clip(im + r.normal(0, 18, im.shape), 0, 255).astype(np.uint8)),
    "jpeg": ("normal", lambda im: cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 12])[1], 0)),
    "bright": ("normal", lambda im: np.clip(im.astype(float) * 1.3 + 25, 0, 255).astype(np.uint8)),  # 과노출
}
if not (BENCH / "labels.csv").exists():
    rng = np.random.default_rng(2031)                                         # 02장 2026, 03장 2027, 04장 2028과 겹치지 않게
    labels = ["ok"] * 30 + [make_parts.DEFECTS[i % 6] for i in range(30)]
    rows = []
    for cond, (light, fx) in CONDITIONS.items():
        (BENCH / cond).mkdir(parents=True, exist_ok=True)
        for i, label in enumerate(labels):
            img = make_parts.make_image(rng, label, light)[0]
            img = fx(img) if fx else img
            cv2.imwrite(str(BENCH / cond / f"{i:03d}_{label}.png"), img)
            rows.append({"light": cond, "file": f"{cond}/{i:03d}_{label}.png", "label": label})
    with open(BENCH / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["light", "file", "label"])
        w.writeheader()
        w.writerows(rows)
with open(BENCH / "labels.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"평가셋 {len(rows)}장: 조건 {len(CONDITIONS)}개 x 60장 (정상 30, 불량 30)")


def load_script(path):
    """앞 장의 스크립트를 한 번 실행해, 그 안의 기준값(spec)과 판정 함수를 그대로 가져온다"""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


# 2. 방법마다 "불량인가"를 판정해 저장
def run_rule():
    m = load_script(ROOT / "ch02_rule_based" / "02_6_rule_inspection.py")      # 2차 규칙 + 조명 보정 (02-6의 최종)
    return lambda path: bool(m.judge(m.measure(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), True), m.spec, True))


def run_cnn():
    import torch
    from torch import nn
    from torchvision import models, transforms
    sys.path.insert(0, str(ROOT / "ch03_deep_learning"))
    from parts_cls import MEAN, STD
    net = models.resnet18()
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(ROOT / "outputs" / "ch03" / "resnet18_parts.pt", map_location="cpu"))
    net = net.cuda().eval()
    norm = transforms.Normalize(MEAN, STD)

    def predict(path):                                                        # 03-5의 PartsDataset과 같은 전처리
        img = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (224, 168), interpolation=cv2.INTER_AREA)
        x = norm(torch.from_numpy(img).float().div(255).unsqueeze(0).repeat(3, 1, 1)).unsqueeze(0).cuda()
        with torch.no_grad():
            return bool(net(x).argmax(1).item() == 1)
    return predict


def run_yolo():
    m = load_script(ROOT / "ch04_detection" / "04_5_evaluate.py")              # 04-4 모델 + 04-5의 판정 규칙
    return lambda path: bool(m.judge(m.features(m.model(path, device=m.device, conf=0.25, verbose=False)[0])))


def run_vlm():
    from vlm import ask
    q = ("This is a photo of a metal bracket on a conveyor belt. A good bracket is a rounded rectangle with exactly two round holes, "
         "no broken corners, no cracks, placed straight in the center. Is this bracket defective? Answer 'ok' or 'defect' only.")
    from PIL import Image
    return lambda path: ask(q, [Image.open(path)], max_tokens=8, base_url=args.vlm_url)["text"].strip().lower().startswith("defect")


RUNNERS = {"rule": run_rule, "cnn": run_cnn, "yolo": run_yolo, "vlm": run_vlm}
for name in args.methods:
    predict = RUNNERS[name]()
    tag = name
    if name == "vlm":
        from vlm import model_name
        tag = "vlm-" + model_name(args.vlm_url).split("/")[-1]
    predict(BENCH / rows[0]["file"])                                          # 워밍업
    t = time.perf_counter()
    preds = {r["file"]: predict(BENCH / r["file"]) for r in rows}
    ms = (time.perf_counter() - t) / len(rows) * 1000
    (OUT / f"bench_{tag}.json").write_text(json.dumps({"ms": ms, "preds": preds}), encoding="utf-8")
    print(f"  {tag}: {len(rows)}장 {ms:.1f}ms/장")

# 3. 표 — 저장된 모든 방법을 같은 기준으로
results = {p.stem[6:]: json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("bench_*.json"))}
conds = list(CONDITIONS)
print(f"\n정확도 (조건별, 각 60장)\n{'방법':<22}" + "".join(f"{c:>9}" for c in conds) + f"{'전체':>8}{'ms/장':>8}")
for tag, res in results.items():
    acc = {c: np.mean([res["preds"][r["file"]] == (r["label"] != "ok") for r in rows if r["light"] == c]) for c in conds}
    total = np.mean([res["preds"][r["file"]] == (r["label"] != "ok") for r in rows])
    print(f"{tag:<22}" + "".join(f"{acc[c]:>9.1%}" for c in conds) + f"{total:>8.1%}{res['ms']:>8.1f}")
print(f"\n불량 유형별 검출 (7개 조건 합계, 유형마다 35장) / 정상 오판 (210장 중)\n{'방법':<22}" +
      "".join(f"{d:>13}" for d in make_parts.DEFECTS) + f"{'정상 오판':>9}")
summary = {}
for tag, res in results.items():
    per = defaultdict(list)
    for r in rows:
        per[r["label"]].append(res["preds"][r["file"]])
    fpr = np.mean(per["ok"])
    tpr = np.mean([p for d in make_parts.DEFECTS for p in per[d]])
    summary[tag] = (tpr, fpr)
    print(f"{tag:<22}" + "".join(f"{np.mean(per[d]):>13.0%}" for d in make_parts.DEFECTS) + f"{fpr:>9.1%}")
print("\n불량이 1%인 현장이라면 — 불량이라고 한 것 중 진짜 불량(Precision) = 재현율x0.01 / (재현율x0.01 + 오판율x0.99)")
for tag, (tpr, fpr) in summary.items():
    prec = tpr * 0.01 / (tpr * 0.01 + fpr * 0.99) if tpr + fpr else 0
    print(f"  {tag:<22} 재현율 {tpr:.1%}, 오판율 {fpr:.1%} → 1% 현장의 Precision {prec:.1%} (평가셋 50%에서는 {tpr / (tpr + fpr) if tpr + fpr else 0:.1%})")
