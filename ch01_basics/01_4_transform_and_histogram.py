"""01-4. 이미지 변환과 밝기 보정 — Resize·Crop·Rotate·Flip·Padding, Histogram·CLAHE

실행:
    python ch01_basics/01_4_transform_and_histogram.py
"""
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ch01"
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(ROOT / "data" / "images" / "bus.jpg"))
h, w = img.shape[:2]
print("원본:", w, "x", h)

# 1. Resize — 줄일 때는 INTER_AREA, 키울 때는 INTER_LINEAR/INTER_CUBIC
for name, interp in [("NEAREST", cv2.INTER_NEAREST), ("LINEAR", cv2.INTER_LINEAR),
                     ("AREA", cv2.INTER_AREA), ("CUBIC", cv2.INTER_CUBIC)]:
    t = time.perf_counter()
    for _ in range(50):
        small = cv2.resize(img, (w // 4, h // 4), interpolation=interp)
    ms = (time.perf_counter() - t) / 50 * 1000
    cv2.imwrite(str(OUT / f"01_4_resize_{name}.png"), cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST))
    print(f"resize 1/4 {name:<7}: {ms:.3f} ms")

# 2. Crop, Flip, Rotate
crop = img[230:730, 0:810]
flip_h = cv2.flip(img, 1)                              # 1: 좌우, 0: 상하, -1: 둘 다
rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)       # 90도 단위는 손실 없이 회전
M = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)   # 임의 각도: 회전 행렬 + warpAffine
rot15 = cv2.warpAffine(img, M, (w, h))
print("\ncrop:", crop.shape, "| rot90:", rot90.shape, "| rot15:", rot15.shape)
print("rot15에서 잘려 검게 된 Pixel 비율:", f"{(rot15.sum(axis=2) == 0).mean():.1%}")

# 3. Padding — 비율을 유지하며 정사각형으로 맞추기(Letterbox, 04장 YOLO 입력에서 다시 나온다)
size = 640
scale = size / max(h, w)
resized = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
pad_w, pad_h = size - resized.shape[1], size - resized.shape[0]
letterbox = cv2.copyMakeBorder(resized, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
print(f"\nletterbox: scale={scale:.4f}, 크기 {resized.shape[1]}x{resized.shape[0]} → {letterbox.shape[1]}x{letterbox.shape[0]}")
cv2.imwrite(str(OUT / "01_4_letterbox.jpg"), letterbox)

# 4. 밝기·대비 — out = alpha * in + beta (0~255로 잘림)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dark = cv2.convertScaleAbs(gray, alpha=0.4, beta=10)    # 어둡고 대비가 낮은 사진을 만든다
for name, g in [("원본", gray), ("어둡게", dark)]:
    print(f"{name:<4} 평균 {g.mean():6.1f} | 표준편차 {g.std():5.1f} | 범위 {g.min()}~{g.max()}")

# 5. Histogram Equalization vs CLAHE
eq = cv2.equalizeHist(dark)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(dark)
for name, g in [("equalizeHist", eq), ("CLAHE", clahe)]:
    print(f"{name:<12} 평균 {g.mean():6.1f} | 표준편차 {g.std():5.1f} | 범위 {g.min()}~{g.max()}")
hist = cv2.calcHist([dark], [0], None, [256], [0, 256]).ravel()
print("어둡게 한 이미지에서 실제로 쓰인 밝기 단계:", int((hist > 0).sum()), "/ 256")
for name, g in [("equalizeHist", eq), ("CLAHE", clahe)]:
    levels = np.unique(g)
    print(f"{name:<12} 결과의 밝기 단계 {len(levels)}개 | 이웃한 단계 사이 간격 최대 {np.diff(levels).max()}")
cv2.imwrite(str(OUT / "01_4_dark_eq_clahe.jpg"), np.hstack([dark, eq, clahe]))
