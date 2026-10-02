"""02-5. Perspective Transform — 번호판 정면화, 바닥면 Bird's-eye View

비스듬히 찍힌 번호판(합성)을 찾아 정면으로 펴고, 원근 때문에 간격이 달라 보이는
바닥 격자를 위에서 내려다본 모습으로 되돌린다.

실행:
    python ch02_rule_based/02_5_perspective.py
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ch02"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

# 1. 번호판 만들기(정면) → 비스듬한 장면에 붙이기
PW, PH = 520, 110
plate = np.full((PH, PW), 245, np.uint8)
cv2.rectangle(plate, (14, 14), (PW - 15, PH - 15), 20, 5)         # 테두리 바깥에 흰 여백을 둔다
cv2.putText(plate, "12A 3456", (52, 80), cv2.FONT_HERSHEY_DUPLEX, 2.2, 20, 5)

scene = np.clip(70 + rng.normal(0, 8, (480, 640)), 0, 255).astype(np.uint8)
true_quad = np.float32([[150, 170], [470, 115], [500, 245], [168, 318]])   # 좌상, 우상, 우하, 좌하
rect = np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]])
H = cv2.getPerspectiveTransform(rect, true_quad)
warped = cv2.warpPerspective(plate, H, (640, 480))
inside = cv2.warpPerspective(np.full_like(plate, 255), H, (640, 480)) > 0
scene[inside] = warped[inside]
cv2.imwrite(str(OUT / "02_5_scene.png"), scene)


# 2. 번호판 꼭짓점 찾기 — 가장 큰 밝은 윤곽을 4각형으로 근사
def order_corners(pts):
    """좌상·우상·우하·좌하 순서로 정렬 (x+y 최소=좌상, 최대=우하, y-x 최소=우상, 최대=좌하)"""
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


_, b = cv2.threshold(cv2.GaussianBlur(scene, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key=cv2.contourArea)
approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
print("근사한 꼭짓점 수:", len(approx))
found = order_corners(approx.reshape(-1, 2).astype(np.float32))
print("꼭짓점 오차(px):", np.linalg.norm(found - true_quad, axis=1).round(2))

# 3. 정면으로 펴기
frontal = cv2.warpPerspective(scene, cv2.getPerspectiveTransform(found, rect), (PW, PH))
diff = cv2.absdiff(frontal, plate)
print(f"정면화 결과와 원본 번호판의 평균 밝기 차이: {diff.mean():.1f} (0~255)")
cv2.imwrite(str(OUT / "02_5_frontal.png"), np.vstack([frontal, plate]))

# 4. Bird's-eye View — 위에서 본 바닥 격자를 비스듬히 찍은 장면으로 만든 뒤 되돌린다
top = np.full((400, 400), 230, np.uint8)
for v in range(20, 400, 40):
    cv2.line(top, (v, 0), (v, 399), 40, 3)
    cv2.line(top, (0, v), (399, v), 40, 3)
top_quad = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
view_quad = np.float32([[230, 120], [410, 120], [600, 460], [40, 460]])          # 카메라에 보이는 사다리꼴
view = cv2.warpPerspective(top, cv2.getPerspectiveTransform(top_quad, view_quad), (640, 480),
                           borderValue=120)


def line_gaps(image, row):
    """주어진 행에서 어두운 선의 위치를 찾아 간격을 잰다"""
    dark = np.where(image[row] < 100)[0]
    centers = [g.mean() for g in np.split(dark, np.where(np.diff(dark) > 1)[0] + 1) if len(g)]
    return np.diff(centers)


far, near = line_gaps(view, 140), line_gaps(view, 440)
print(f"\n원근 장면: 먼 쪽 선 간격 평균 {far.mean():.1f}px, 가까운 쪽 {near.mean():.1f}px (비율 {near.mean() / far.mean():.2f})")
birdseye = cv2.warpPerspective(view, cv2.getPerspectiveTransform(view_quad, top_quad), (400, 400))
far2, near2 = line_gaps(birdseye, 30), line_gaps(birdseye, 370)
print(f"Bird's-eye: 먼 쪽 {far2.mean():.1f}px, 가까운 쪽 {near2.mean():.1f}px (비율 {near2.mean() / far2.mean():.2f}, 실제 40px)")
cv2.imwrite(str(OUT / "02_5_view.png"), view)
cv2.imwrite(str(OUT / "02_5_birdseye.png"), birdseye)
