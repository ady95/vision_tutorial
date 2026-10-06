"""16-1 프로젝트 2단계 — 결함 영역 분할 모델(YOLO26n-seg) 학습

make_data.py가 만든 data/parts_seg (train 600, val 150)로 학습한다. 결과: outputs/ch16/p1/seg/weights/best.pt

실행 (저장소 루트에서):
    python ch16_projects/p1_defect_inspection/train_seg.py
"""
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
t = time.perf_counter()
model = YOLO("yolo26n-seg.pt")                                                # COCO로 사전학습한 분할 모델에서 시작 (04-4와 같은 미세조정)
model.train(data=str(ROOT / "data" / "parts_seg" / "data.yaml"), epochs=50, imgsz=640, batch=16, seed=0, deterministic=True,
            device=0 if torch.cuda.is_available() else "cpu", project=str(ROOT / "outputs" / "ch16" / "p1"), name="seg",
            exist_ok=True, plots=False, verbose=False)
m = model.val(data=str(ROOT / "data" / "parts_seg" / "data.yaml"), split="val", verbose=False, plots=False)
print(f"학습 {time.perf_counter() - t:.0f}초 | 검증 마스크 mAP50 {m.seg.map50:.3f}, mAP50-95 {m.seg.map:.3f} | 박스 mAP50 {m.box.map50:.3f}")
for i, n in enumerate(m.names.values()):
    print(f"  {n:<13} 마스크 AP50 {m.seg.ap50[i]:.3f}")
