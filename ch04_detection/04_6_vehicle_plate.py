"""04-6. 실습: 차량과 번호판 검출 — 사전학습 YOLO로 차량을 찾고, 그 안에서 Rule(02장)로 번호판을 찾는다

COCO에는 '번호판' 클래스가 없다. 그래서 차량 박스 안에서 02장의 방법(Sobel, Morphology, Contour)으로 찾는다.
결과 이미지는 개인정보 보호를 위해 번호판 영역을 모자이크해 저장한다.

실행 (저장소 루트에서):
    python ch04_detection/04_6_vehicle_plate.py --images 차량_사진_폴더
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ch04" / "plates"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--images", default=str(ROOT / "data" / "images"))
args = ap.parse_args()
device = 0 if torch.cuda.is_available() else "cpu"
VEHICLES = {2: "car", 5: "bus", 7: "truck"}                  # COCO 클래스 번호
PW, PH = 520, 110                                           # 정면화할 번호판 크기 (02-5와 같음)


def order_corners(pts):
    """좌상·우상·우하·좌하 순서로 정렬 (02-5)"""
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def find_plate(roi):
    """차량 영역에서 번호판 후보를 찾는다: 글자 때문에 세로 경계가 촘촘하고, 밝고, 가로로 긴 사각형"""
    scale = 400 / roi.shape[1]                              # 차량 크기에 상관없이 같은 커널을 쓰도록 폭 400으로 맞춘다
    small = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    _, edges = cv2.threshold(gx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blob = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    best = None
    for c in contours:
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        long_, short = max(rw, rh), min(rw, rh)
        if short < 8 or not 2.0 <= long_ / short <= 6.5:   # 가로세로 비: 신형 4.7, 구형 약 2
            continue
        if not 0.12 <= long_ / w <= 0.45 or cy < 0.3 * h:   # 번호판 폭은 차량 폭의 12~45%, 차량 아래쪽에 있다
            continue
        box = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.int32)
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [box], 255)
        bright = cv2.mean(gray, mask)[0]
        density = cv2.mean(edges, mask)[0] / 255
        if bright < 100:                                     # 흰색·노란색 바탕만 (어두운 그릴 제외)
            continue
        score = density * bright / 255 / (1 + abs(long_ / short - 4.7) / 4.7)
        if best is None or score > best[0]:
            best = (score, cv2.boxPoints(((cx, cy), (rw, rh), ang)) / scale, long_ / short)
    return best


model = YOLO("yolo26s.pt")
files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".png"))
total_vehicles = total_plates = 0
for f in files:
    img = cv2.imread(str(f))
    r = model(img, device=device, classes=list(VEHICLES), conf=0.4, verbose=False)[0]
    vis = img.copy()
    found = []
    for b, c, s in zip(r.boxes.xyxy.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
        x1, y1, x2, y2 = b
        if (x2 - x1) < 150:                                  # 너무 작은 차량은 번호판을 읽을 수 없으니 건너뛴다
            continue
        total_vehicles += 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        plate = find_plate(img[y1:y2, x1:x2])
        if plate is None:
            found.append(f"{VEHICLES[c]} {s:.2f} 번호판 못 찾음")
            continue
        total_plates += 1
        quad = order_corners(plate[1] + [x1, y1])
        frontal = cv2.warpPerspective(img, cv2.getPerspectiveTransform(
            quad, np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]])), (PW, PH))
        cv2.imwrite(str(OUT / f"{f.stem}_plate{len(found)}.png"), frontal)          # 16-2에서 OCR 입력으로 쓴다
        qx1, qy1 = quad.min(axis=0).astype(int)
        qx2, qy2 = quad.max(axis=0).astype(int)
        patch = vis[qy1:qy2, qx1:qx2]                         # 결과 그림에서는 번호판을 모자이크
        vis[qy1:qy2, qx1:qx2] = cv2.resize(cv2.resize(patch, (8, 3)), patch.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 0, 255), 3)
        found.append(f"{VEHICLES[c]} {s:.2f} 번호판 {qx2 - qx1}x{qy2 - qy1}px 비율 {plate[2]:.1f}")
    cv2.imwrite(str(OUT / f"{f.stem}_vis.jpg"), vis)
    print(f"{f.name}: " + (" | ".join(found) if found else "차량 없음"))
print(f"\n차량 {total_vehicles}대 중 번호판 후보를 찾은 차량 {total_plates}대")
