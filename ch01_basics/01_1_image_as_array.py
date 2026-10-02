"""01-1. 컴퓨터는 이미지를 어떻게 보는가 — 이미지는 NumPy 배열이다

실행:
    python ch01_basics/01_1_image_as_array.py
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ch01"
OUT.mkdir(parents=True, exist_ok=True)

# 1. 이미지를 읽으면 NumPy 배열이 된다
img = cv2.imread(str(ROOT / "data" / "images" / "bus.jpg"))
print("type :", type(img))
print("shape:", img.shape, "(height, width, channel)")
print("dtype:", img.dtype, "| 최솟값", img.min(), "| 최댓값", img.max())
print("메모리:", img.nbytes, "bytes =", img.shape[0], "x", img.shape[1], "x", img.shape[2])

# 2. Pixel 읽기 — 인덱스 순서는 [y, x] (행, 열)
h, w = img.shape[:2]
x, y = 520, 560  # 버스 옆면(파란색)의 한 점
print(f"\n(x={x}, y={y}) 의 값 [B, G, R]:", img[y, x])

# 3. Pixel 바꾸기와 영역 선택(ROI) — 슬라이싱은 복사가 아니라 같은 메모리를 가리킨다
roi = img[300:700, 380:780]           # y1:y2, x1:x2
print("ROI shape:", roi.shape)
roi_copy = roi.copy()
roi[:20, :] = (0, 0, 255)             # ROI의 위쪽 20줄을 빨간색(BGR)으로
print("원본도 바뀌었는가:", (img[300, 380] == (0, 0, 255)).all())
cv2.imwrite(str(OUT / "01_1_roi_view.jpg"), img)
cv2.imwrite(str(OUT / "01_1_roi_copy.jpg"), roi_copy)

# 4. Bounding Box 표기법 변환
box_xyxy = np.array([380, 300, 780, 700], dtype=float)   # x1, y1, x2, y2


def xyxy_to_xywh(b):
    """좌상단·우하단 좌표 → 중심점·너비·높이 (YOLO 방식)"""
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


def normalize(b, width, height):
    """픽셀 좌표 → 0~1 정규화 좌표"""
    return b / np.array([width, height, width, height])


box_xywh = xyxy_to_xywh(box_xyxy)
print("\nxyxy           :", box_xyxy)
print("xywh (중심 기준):", box_xywh)
print("xywh 정규화     :", normalize(box_xywh, w, h).round(4))
