"""04-3. Ultralytics YOLO 시작하기 — 모델 크기 비교, 결과 객체 읽기, 영상 추론

실행 (저장소 루트에서):
    uv pip install -e ".[dl]"
    python ch04_detection/04_3_yolo_inference.py
    python ch04_detection/04_3_yolo_inference.py --video data/own_video/cctv.mp4   # 다른 영상으로 (선택)
"""
import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "data" / "images" / "bus.jpg"
OUT = ROOT / "outputs" / "ch04"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--video", default=str(ROOT / "outputs" / "ch01" / "01_2_pan.mp4"), help="01-2에서 만든 영상이 기본값")
args = ap.parse_args()
device = 0 if torch.cuda.is_available() else "cpu"

# 1. 모델 크기별 비교 — 처음 실행하면 가중치를 자동으로 내려받는다
print(f"{'모델':<12}{'파라미터':>12}{'파일':>9}{'추론(ms)':>10}  검출 결과")
for name in ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"]:
    model = YOLO(name)
    params = sum(p.numel() for p in model.model.parameters())
    model(IMG, device=device, verbose=False)                      # 워밍업
    times = []
    for _ in range(10):
        r = model(IMG, device=device, verbose=False)[0]
        times.append(r.speed["inference"])
    found = {}
    for c in r.boxes.cls.int().tolist():
        found[r.names[c]] = found.get(r.names[c], 0) + 1
    size = Path(name).stat().st_size / 1024 ** 2
    print(f"{name:<12}{params:>12,}{size:>7.1f}MB{sorted(times)[5]:>10.1f}  {found}")

# 2. 결과 객체 읽기 — 박스·신뢰도·클래스
model = YOLO("yolo26s.pt")
r = model(IMG, device=device, verbose=False)[0]
print(f"\n이미지 크기 {r.orig_shape} | 전처리 {r.speed['preprocess']:.1f}ms, 추론 {r.speed['inference']:.1f}ms, "
      f"후처리 {r.speed['postprocess']:.1f}ms")
print(f"{'class':<10}{'conf':>6}  {'xyxy (픽셀)':<28}{'xywhn (정규화 중심·크기)'}")
for box in r.boxes:
    cls = r.names[int(box.cls)]
    xyxy = [round(v) for v in box.xyxy[0].tolist()]
    xywhn = [round(v, 3) for v in box.xywhn[0].tolist()]
    print(f"{cls:<10}{float(box.conf):>6.2f}  {str(xyxy):<28}{xywhn}")
cv2.imwrite(str(OUT / "04_3_bus_yolo26s.jpg"), r.plot())               # 박스를 그린 이미지 (BGR)

# 3. 신뢰도 기준(conf)을 바꾸면
for conf in [0.1, 0.25, 0.5, 0.8]:
    r = model(IMG, device=device, conf=conf, verbose=False)[0]
    print(f"conf >= {conf:<4}: 검출 {len(r.boxes)}개")

# 4. 영상 추론 — stream=True로 한 프레임씩 받는다
cap = cv2.VideoCapture(args.video)
fps_in = cap.get(cv2.CAP_PROP_FPS)
cap.release()
n, t0, counts = 0, time.perf_counter(), []
for r in model(args.video, device=device, stream=True, verbose=False):
    n += 1
    counts.append(len(r.boxes))
sec = time.perf_counter() - t0
print(f"\n영상 {Path(args.video).name}: {n}프레임 ({fps_in:.0f}fps 영상) | {sec:.1f}초, 초당 {n / sec:.1f}프레임 처리 "
      f"| 프레임당 검출 평균 {sum(counts) / n:.1f}개")
