"""05-2. 실습: YOLO Segmentation — 자동차 영역 분할하기

박스와 마스크를 비교하고, 마스크로 면적·외곽선·배경 제거를 해 본다.

실행 (저장소 루트에서):
    python ch05_seg_pose_track/05_2_yolo_seg.py                       # data/images
    python ch05_seg_pose_track/05_2_yolo_seg.py --images 차량_사진_폴더
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ch05"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--images", default=str(ROOT / "data" / "images"))
args = ap.parse_args()
device = 0 if torch.cuda.is_available() else "cpu"

# 1. 검출 모델과 분할 모델 — 이름 뒤에 -seg가 붙는다
for name in ["yolo26s.pt", "yolo26s-seg.pt"]:
    m = YOLO(name)
    m(ROOT / "data" / "images" / "bus.jpg", device=device, verbose=False)            # 워밍업
    t = time.perf_counter()
    for _ in range(10):
        r = m(ROOT / "data" / "images" / "bus.jpg", device=device, verbose=False)[0]
    ms = (time.perf_counter() - t) * 100
    print(f"{name:<16} 파라미터 {sum(p.numel() for p in m.model.parameters()):>11,} | 1장 {ms:5.1f} ms (전후처리 포함)"
          f" | 마스크 {'있음' if r.masks is not None else '없음'}")

# 2. 결과 객체의 마스크 — 두 가지 형태
model = YOLO("yolo26s-seg.pt")
r = model(ROOT / "data" / "images" / "bus.jpg", device=device, verbose=False)[0]
print(f"\nmasks.data: {tuple(r.masks.data.shape)} (물체 수 x 모델 입력 해상도의 마스크)")
print(f"masks.xy[0]: 외곽선 점 {len(r.masks.xy[0])}개 (원본 이미지 좌표의 다각형)")

# 3. 박스와 마스크 — 박스 안에서 물체가 실제로 차지하는 비율
files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".png"))
print(f"\n{'파일':<14}{'클래스':<11}{'conf':>5}{'박스 넓이':>11}{'마스크 넓이':>12}{'마스크/박스':>11}")
for f in files:
    img = cv2.imread(str(f))
    r = model(img, device=device, verbose=False)[0]
    if r.masks is None:
        continue
    vis = img.copy()
    for box, cls, conf, poly in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.masks.xy):
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)                 # 원본 해상도 마스크
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        print(f"{f.stem[:13]:<14}{r.names[cls]:<11}{conf:>5.2f}{box_area:>11,.0f}{int((mask > 0).sum()):>12,}"
              f"{(mask > 0).sum() / box_area:>11.1%}")
        cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 255, 255), 3)
    cv2.imwrite(str(OUT / f"05_2_{f.stem}_seg.jpg"), vis)

# 4. 마스크로 배경 지우기 — 가장 큰 차량 하나만 남긴다
for f in files:
    img = cv2.imread(str(f))
    r = model(img, device=device, classes=[2, 5, 7], verbose=False)[0]
    if r.masks is None:
        continue
    areas = [cv2.contourArea(p.astype(np.float32)) for p in r.masks.xy]
    poly = r.masks.xy[int(np.argmax(areas))].astype(np.int32)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    cut = cv2.bitwise_and(img, img, mask=mask)
    cut[mask == 0] = 255                                                 # 배경을 흰색으로
    x, y, w, h = cv2.boundingRect(poly)
    cv2.imwrite(str(OUT / f"05_2_{f.stem}_cutout.png"), cut[y:y + h, x:x + w])
    print(f"{f.stem}: 가장 큰 차량 외곽선 길이 {cv2.arcLength(poly, True):,.0f}px, 면적 {max(areas):,.0f}px²")
