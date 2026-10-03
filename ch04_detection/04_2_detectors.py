"""04-2. Detector의 발전 — Two-stage(Faster R-CNN), 초기 One-stage(SSD), YOLO26을 같은 이미지로 비교

세 모델 모두 COCO 80종으로 학습된 공개 가중치를 쓴다 (torchvision 모델은 처음 실행할 때 자동으로 내려받는다).

실행 (저장소 루트에서):
    python ch04_detection/04_2_detectors.py
    python ch04_detection/04_2_detectors.py --image 다른_이미지.jpg
"""
import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from torchvision.models import detection as tvd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--image", default=str(ROOT / "data" / "images" / "bus.jpg"))
ap.add_argument("--conf", type=float, default=0.5)
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
img = cv2.imread(args.image)
rgb = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)


def timed(fn, n=10):
    fn()                                                  # 워밍업
    times = []
    for _ in range(n):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return out, sorted(times)[n // 2]                     # 중앙값


rows = []
for name, builder, weights in [
    ("Faster R-CNN (Two-stage, 2015)", tvd.fasterrcnn_resnet50_fpn_v2, tvd.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
    ("SSD300 (One-stage, 2016)", tvd.ssd300_vgg16, tvd.SSD300_VGG16_Weights.COCO_V1),
]:
    model = builder(weights=weights).eval().to(device)
    names = weights.meta["categories"]
    with torch.no_grad():
        out, ms = timed(lambda: model([rgb.to(device)])[0])
    found = Counter(names[int(l)] for l, s in zip(out["labels"], out["scores"]) if s >= args.conf)
    rows.append((name, sum(p.numel() for p in model.parameters()), ms, dict(found)))

yolo = YOLO("yolo26s.pt")
r, ms = timed(lambda: yolo(img, device=0 if device == "cuda" else "cpu", conf=args.conf, verbose=False)[0])
rows.append(("YOLO26s (One-stage, 2025)", sum(p.numel() for p in yolo.model.parameters()), ms,
             dict(Counter(r.names[int(c)] for c in r.boxes.cls))))

print(f"이미지 {Path(args.image).name} {img.shape[1]}x{img.shape[0]} | {device} | 신뢰도 {args.conf} 이상")
print(f"{'모델':<32}{'파라미터':>13}{'1장(ms)':>9}  검출")
for name, params, ms, found in rows:
    print(f"{name:<32}{params:>13,}{ms:>9.1f}  {found}")
