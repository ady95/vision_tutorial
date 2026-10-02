"""01-3. Color Space 이해하기 — BGR, Grayscale, HSV, LAB

실행:
    python ch01_basics/01_3_color_space.py
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ch01"
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(ROOT / "data" / "images" / "bus.jpg"))
x, y = 520, 560                                       # 버스 옆면(파란색)

# 1. 같은 Pixel을 네 가지 Color Space로 보기
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
print("BGR :", img[y, x])
print("RGB :", img[y, x][::-1])
print("Gray:", gray[y, x])
print("HSV :", hsv[y, x], "(H는 0~179, S·V는 0~255)")
print("LAB :", lab[y, x])

# 2. HSV로 파란색 영역만 골라내기 — H(색상)로 색을 고르고 S·V로 너무 흐리거나 어두운 곳을 뺀다
lower, upper = np.array([95, 120, 70]), np.array([125, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
blue_only = cv2.bitwise_and(img, img, mask=mask)
print(f"\n파란색 Pixel 비율: {mask.mean() / 255:.1%}")
cv2.imwrite(str(OUT / "01_3_blue_mask.png"), mask)
cv2.imwrite(str(OUT / "01_3_blue_only.jpg"), blue_only)

# 3. 조명이 절반으로 어두워지면? — BGR 범위와 HSV 범위를 같은 조건에서 비교한다
darker = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
hsv_dark = cv2.cvtColor(darker, cv2.COLOR_BGR2HSV)
print("\n어두워진 같은 Pixel")
print("  BGR:", darker[y, x], "<- 원래", img[y, x])
print("  HSV:", hsv_dark[y, x], "<- 원래", hsv[y, x], "(H·S는 거의 그대로, V만 절반)")

bgr_lower, bgr_upper = np.array([120, 50, 0]), np.array([255, 150, 90])   # 밝은 사진에 맞춘 BGR 범위
hsv_lower_dark = np.array([95, 120, 35])                                  # HSV는 V 하한만 낮춘다
for name, m_bright, m_dark in [
    ("BGR 범위", cv2.inRange(img, bgr_lower, bgr_upper), cv2.inRange(darker, bgr_lower, bgr_upper)),
    ("HSV 범위", mask, cv2.inRange(hsv_dark, hsv_lower_dark, upper)),
]:
    print(f"  {name}: 밝은 사진 {m_bright.mean() / 255:.1%} → 어두운 사진 {m_dark.mean() / 255:.1%}")

# 4. Grayscale 변환 공식 확인 — 단순 평균이 아니라 가중합(0.299R + 0.587G + 0.114B)
b, g, r = img[y, x].astype(float)
print(f"\n가중합: {0.299 * r + 0.587 * g + 0.114 * b:.1f} | 단순 평균: {(r + g + b) / 3:.1f} | OpenCV: {gray[y, x]}")
