"""04-4. Custom Dataset으로 YOLO 학습하기 — 합성 부품의 "불량이 어디에 있는가" 검출기

사전학습 가중치에서 시작한 미세조정(Fine-tuning)과, 같은 구조를 처음부터 학습한 경우를 비교한다.

실행 (저장소 루트에서):
    python data/make_parts_det.py
    python ch04_detection/04_4_train.py
"""
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "parts_det" / "data.yaml"
RUNS = ROOT / "outputs" / "ch04" / "runs"
device = 0 if torch.cuda.is_available() else "cpu"

results = {}
for name, start in [("finetune", "yolo26n.pt"), ("scratch", "yolo26n.yaml")]:
    model = YOLO(start)                   # .pt: COCO 사전학습 가중치 / .yaml: 구조만 (무작위 초기화)
    t = time.perf_counter()
    model.train(data=str(DATA), epochs=50, imgsz=640, batch=16, device=device, seed=0, deterministic=True,
                project=str(RUNS), name=name, exist_ok=True, verbose=False, plots=False)
    minutes = (time.perf_counter() - t) / 60
    best = YOLO(RUNS / name / "weights" / "best.pt")
    m = best.val(data=str(DATA), split="test", device=device, verbose=False, plots=False)
    results[name] = (minutes, m)

print(f"\n{'':<10}{'학습(분)':>9}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}   클래스별 AP50 (part / hole / chip / crack)")
for name, (minutes, m) in results.items():
    b = m.box
    per_class = " / ".join(f"{v:.3f}" for v in b.ap50)
    print(f"{name:<10}{minutes:>9.1f}{b.mp:>7.3f}{b.mr:>7.3f}{b.map50:>8.3f}{b.map:>10.3f}   {per_class}")
print(f"\n가중치: {RUNS / 'finetune' / 'weights' / 'best.pt'}")
