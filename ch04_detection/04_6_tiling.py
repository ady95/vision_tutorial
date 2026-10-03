"""04-6. Small Object Detection — 입력 해상도와 Tiling

큰 이미지를 640으로 줄여 한 번에 검출할 때, 해상도를 키울 때, 640 크기 조각(Tile)으로 나눠 검출할 때를 비교한다.
결과 그림에서는 차량 박스 아래쪽(번호판이 있을 법한 곳)을 모자이크한다.

실행 (저장소 루트에서):
    python ch04_detection/04_6_tiling.py --image 큰_이미지.jpg
"""
import argparse
import time
from pathlib import Path

import cv2
import torch
import torchvision
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ch04"
ap = argparse.ArgumentParser()
ap.add_argument("--image", default=str(ROOT / "data" / "images" / "bus.jpg"))
args = ap.parse_args()
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("yolo26s.pt")
VEHICLES = [2, 3, 5, 7]                                      # car, motorcycle, bus, truck
img = cv2.imread(args.image)
H, W = img.shape[:2]


def detect(image, imgsz):
    r = model(image, device=device, imgsz=imgsz, classes=VEHICLES, conf=0.3, verbose=False)[0]
    return r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu()


def detect_tiled(tile=640, overlap=0.25):
    """이미지를 겹치는 조각으로 나눠 원래 해상도 그대로 검출하고, 조각 경계의 중복은 NMS로 합친다"""
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(W - tile, 0) + 1, step)) + ([W - tile] if (W - tile) % step else [])
    ys = list(range(0, max(H - tile, 0) + 1, step)) + ([H - tile] if (H - tile) % step else [])
    boxes, scores, classes = [], [], []
    for y in ys:
        for x in xs:
            b, s, c = detect(img[y:y + tile, x:x + tile], tile)
            boxes.append(b + torch.tensor([x, y, x, y]))
            scores.append(s)
            classes.append(c)
    b, s, c = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
    keep = torchvision.ops.batched_nms(b, s, c, 0.5)
    return b[keep], s[keep], c[keep], len(xs) * len(ys)


def fragments(b):
    """다른 박스 안에 60% 이상 들어가 있는 박스 — 조각 경계에서 차량 하나가 둘로 잘려 생긴 경우가 많다"""
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.max(b[:, None, :2], b[None, :, :2])
    rb = torch.min(b[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=2)
    inside = inter / area[:, None]                           # [i, j]: 박스 i가 박스 j 안에 들어간 비율
    inside.fill_diagonal_(0)
    return (inside.max(dim=1).values > 0.6) & (area < area.max())


def detect_tiled_merged():
    """조각 결과에서, 더 큰 박스 안에 대부분 들어간 작은 박스를 지운다"""
    b, s, c, n = detect_tiled()
    keep = ~fragments(b)
    return b[keep], s[keep], c[keep], n


print(f"이미지 {W}x{H}")
results = {}
for name, fn in [("한 번에 640", lambda: detect(img, 640) + (1,)),
                 ("한 번에 1280", lambda: detect(img, 1280) + (1,)),
                 ("조각 640 (25% 겹침)", detect_tiled),
                 ("조각 + 잘린 박스 합치기", detect_tiled_merged)]:
    fn()                                                     # 워밍업
    t = time.perf_counter()
    b, s, c, n = fn()
    ms = (time.perf_counter() - t) * 1000
    widths = (b[:, 2] - b[:, 0]).tolist()
    results[name] = b
    print(f"{name:<18}: 박스 {len(b):2d}개 | 그중 조각 {int(fragments(b).sum()):2d}개 | 폭 60px 미만 "
          f"{sum(w < 60 for w in widths):2d}개 | 가장 작은 폭 {min(widths):4.0f}px | 추론 {n}회 {ms:6.1f} ms")

for name, b in results.items():
    vis = img.copy()
    for x1, y1, x2, y2 in b.int().tolist():
        py1 = y1 + int((y2 - y1) * 0.5)                      # 번호판이 있을 법한 아래쪽 절반을 모자이크
        patch = vis[py1:y2, x1:x2]
        if patch.size:
            vis[py1:y2, x1:x2] = cv2.resize(cv2.resize(patch, (max(1, (x2 - x1) // 12), max(1, (y2 - py1) // 12))),
                                            (x2 - x1, y2 - py1), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite(str(OUT / f"04_6_tiling_{list(results).index(name)}.jpg"), vis)
