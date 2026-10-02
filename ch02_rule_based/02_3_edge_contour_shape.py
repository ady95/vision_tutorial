"""02-3. Edge, Contour, Shape 분석

실행:
    python data/make_parts.py
    python ch02_rule_based/02_3_edge_contour_shape.py
"""
import csv
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "data" / "parts"
OUT = ROOT / "outputs" / "ch02"
OUT.mkdir(parents=True, exist_ok=True)

name = "000_ok.png"
img = cv2.imread(str(PARTS / "normal" / name), cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5, 5), 0)

# 1. Edge — Gradient(Sobel), Laplacian, Canny
gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
mag = cv2.magnitude(gx, gy)
lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
canny = cv2.Canny(blur, 50, 150)
print(f"Sobel 크기 최댓값 {mag.max():.0f} | Laplacian 범위 {lap.min():.0f}~{lap.max():.0f}")
print(f"Canny Edge Pixel: {(canny > 0).sum()}개 ({(canny > 0).mean():.2%})")
for lo, hi in [(10, 30), (50, 150), (200, 400)]:
    print(f"  Canny({lo}, {hi}): {(cv2.Canny(blur, lo, hi) > 0).sum():6d} Pixel")
cv2.imwrite(str(OUT / "02_3_canny.png"), canny)

# 2. Contour — RETR_CCOMP는 바깥 윤곽과 구멍을 부모·자식 2단계로 돌려준다
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]                        # [다음, 이전, 첫 자식, 부모]
outer = max((i for i in range(len(contours)) if hierarchy[i][3] == -1), key=lambda i: cv2.contourArea(contours[i]))
holes = [i for i in range(len(contours)) if hierarchy[i][3] == outer]
print(f"\n윤곽 {len(contours)}개 = 바깥 1개 + 구멍 {len(holes)}개")

c = contours[outer]
area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
M = cv2.moments(c)
cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
angle = 0.5 * math.degrees(math.atan2(2 * M["mu11"], M["mu20"] - M["mu02"]))
hull_area = cv2.contourArea(cv2.convexHull(c))
print(f"바깥 윤곽: 면적 {area:.0f} | 둘레 {perim:.1f} | 중심 ({cx:.1f}, {cy:.1f}) | 각도 {angle:.2f}도")
print(f"           Solidity(면적/볼록껍질) {area / hull_area:.3f} | 꼭짓점 {len(cv2.approxPolyDP(c, 0.02 * perim, True))}개")

with open(PARTS / "labels.csv", encoding="utf-8") as f:
    truth = next(r for r in csv.DictReader(f) if r["file"] == f"normal/{name}")
print(f"           (정답: 중심 ({truth['cx']}, {truth['cy']}), 각도 {truth['angle']}도)")

for i in holes:
    a, p = cv2.contourArea(contours[i]), cv2.arcLength(contours[i], True)
    (hx, hy), r = cv2.minEnclosingCircle(contours[i])
    print(f"구멍: 중심 ({hx:.0f}, {hy:.0f}) 반지름 {r:.1f} | Circularity 4πA/P² = {4 * math.pi * a / p ** 2:.3f}")

# 3. Connected Component — 개수 세기
multi = cv2.imread(str(PARTS / "multi.png"), cv2.IMREAD_GRAYSCALE)
_, mb = cv2.threshold(multi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
n, labels, stats, centroids = cv2.connectedComponentsWithStats(mb)
areas = stats[1:, cv2.CC_STAT_AREA]
print(f"\nmulti.png: 연결 영역 {n - 1}개, 면적 {sorted(areas.tolist())}")
print(f"면적 100 이상만 세면: {(areas >= 100).sum()}개")

# 4. Hough Transform — 구멍(원)과 가장자리(직선)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=120, param2=25, minRadius=12, maxRadius=30)
found = [] if circles is None else [[round(float(v), 1) for v in c] for c in circles[0]]
print(f"\nHoughCircles: {len(found)}개 [x, y, 반지름] {found}")
lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=60, minLineLength=80, maxLineGap=5)
lines = lines.reshape(-1, 4)                    # OpenCV 4.x는 (N, 1, 4), 5.x는 (N, 4)로 돌려준다
longest = max(lines, key=lambda l: math.hypot(l[2] - l[0], l[3] - l[1]))
x1, y1, x2, y2 = longest
line_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
print(f"HoughLinesP: 직선 {len(lines)}개, 가장 긴 직선 각도 {line_angle:.2f}도")

vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
cv2.circle(vis, (round(cx), round(cy)), 4, (0, 0, 255), -1)
if circles is not None:
    for x, y, r in circles[0]:
        cv2.circle(vis, (round(x), round(y)), round(r), (255, 0, 255), 1)
cv2.imwrite(str(OUT / "02_3_contours.png"), vis)
