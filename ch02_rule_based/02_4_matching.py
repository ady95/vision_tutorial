"""02-4. Template Matching과 Feature Matching

같은 대상을 크기·회전이 바뀐 이미지에서 찾을 때 두 방식이 어떻게 다른지 비교한다.

실행:
    python ch02_rule_based/02_4_matching.py
"""
import math
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ch02"
OUT.mkdir(parents=True, exist_ok=True)

gray = cv2.imread(str(ROOT / "data" / "images" / "bus.jpg"), cv2.IMREAD_GRAYSCALE)
h, w = gray.shape
tx, ty, tw, th = 380, 520, 130, 150             # 버스 옆면의 "e" 로고
template = gray[ty:ty + th, tx:tx + tw]
cv2.imwrite(str(OUT / "02_4_template.png"), template)


def match(image, tmpl):
    res = cv2.matchTemplate(image, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return score, loc


def rotate(image, deg, scale):
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, scale)
    return cv2.warpAffine(image, M, (w, h)), M


# 1. Template Matching — 원본, 축소, 회전
print("Template Matching (TM_CCOEFF_NORMED, 1.0이 완전 일치)")
score, loc = match(gray, template)
print(f"  원본          : 점수 {score:.3f}, 위치 {loc} (정답 {(tx, ty)})")
small = cv2.resize(gray, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
print(f"  0.7배 축소    : 점수 {match(small, template)[0]:.3f}")
rot30, _ = rotate(gray, 30, 1.0)
print(f"  30도 회전     : 점수 {match(rot30, template)[0]:.3f}")

t = time.perf_counter()
best = max((match(small, cv2.resize(template, None, fx=s, fy=s))[0], s) for s in np.linspace(0.5, 1.2, 15))
print(f"  0.7배 + 크기 15단계 탐색: 최고 점수 {best[0]:.3f} (템플릿 {best[1]:.2f}배), {(time.perf_counter() - t) * 1000:.0f} ms")

# 2. Feature Matching — 30도 회전 + 0.7배 축소된 이미지에서 찾기
target, M_true = rotate(gray, 30, 0.7)
print("\nFeature Matching (대상: 30도 회전 + 0.7배)")
for name, detector, norm in [("ORB", cv2.ORB_create(nfeatures=2000), cv2.NORM_HAMMING),
                             ("SIFT", cv2.SIFT_create(), cv2.NORM_L2)]:
    detector.detectAndCompute(gray, None)           # 첫 호출은 초기화 시간이 섞이므로 한 번 돌리고 잰다
    t = time.perf_counter()
    kp1, des1 = detector.detectAndCompute(gray, None)
    kp2, des2 = detector.detectAndCompute(target, None)
    pairs = cv2.BFMatcher(norm).knnMatch(des1, des2, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]    # Lowe의 ratio test
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    H, inlier = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    ms = (time.perf_counter() - t) * 1000
    # 추정한 변환에서 회전·배율을 다시 계산해 정답(30도, 0.7배)과 비교
    est_scale = math.sqrt(abs(np.linalg.det(H[:2, :2])))
    est_angle = -math.degrees(math.atan2(H[1, 0], H[0, 0]))
    print(f"  {name:<4}: 특징점 {len(kp1)}/{len(kp2)} | 좋은 매칭 {len(good)} | RANSAC inlier {int(inlier.sum())}"
          f" | 추정 회전 {est_angle:.2f}도, 배율 {est_scale:.3f} | {ms:.0f} ms")

    # 로고 위치를 추정 변환으로 옮겨 정답 위치와 비교
    corners = np.float32([[tx, ty], [tx + tw, ty], [tx + tw, ty + th], [tx, ty + th]]).reshape(-1, 1, 2)
    est = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    true = cv2.transform(corners, M_true).reshape(-1, 2)
    print(f"        로고 꼭짓점 평균 오차 {np.linalg.norm(est - true, axis=1).mean():.2f} px")

vis = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
cv2.polylines(vis, [est.astype(np.int32)], True, (0, 255, 0), 3)
cv2.imwrite(str(OUT / "02_4_feature_found.jpg"), vis)
