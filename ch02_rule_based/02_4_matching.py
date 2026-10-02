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
logo_mask = np.zeros_like(gray)
logo_mask[ty:ty + th, tx:tx + tw] = 255
corners = np.float32([[tx, ty], [tx + tw, ty], [tx + tw, ty + th], [tx, ty + th]]).reshape(-1, 1, 2)
true_corners = cv2.transform(corners, M_true).reshape(-1, 2)


def feature_match(detector, norm, mask):
    """원본(mask 영역)의 특징점을 대상 이미지에서 찾아 로고 위치·회전·배율을 추정한다"""
    detector.detectAndCompute(gray, None)           # 첫 호출은 초기화 시간이 섞이므로 한 번 돌리고 잰다
    t = time.perf_counter()
    kp1, des1 = detector.detectAndCompute(gray, mask)
    kp2, des2 = detector.detectAndCompute(target, None)
    pairs = cv2.BFMatcher(norm).knnMatch(des1, des2, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]    # Lowe의 ratio test
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    H, inlier = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)          # 로고 위치(원근까지 허용)
    A, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)    # 회전·배율·이동만 허용
    ms = (time.perf_counter() - t) * 1000
    found = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    found_sim = cv2.transform(corners, A).reshape(-1, 2)
    return {
        "kp": f"{len(kp1)}/{len(kp2)}", "good": len(good), "inlier": int(inlier.sum()),
        "angle": -math.degrees(math.atan2(A[1, 0], A[0, 0])), "scale": math.hypot(A[0, 0], A[1, 0]),
        "error": np.linalg.norm(found - true_corners, axis=1).mean(),
        "error_sim": np.linalg.norm(found_sim - true_corners, axis=1).mean(), "ms": ms, "found": found,
    }


print("\nFeature Matching (대상: 30도 회전 + 0.7배)")
print("  오차 = 정답 로고 꼭짓점과의 평균 거리(px), H: Homography, 닮음: 회전·배율·이동만 허용한 변환")
for region, mask in [("로고 영역만", logo_mask), ("이미지 전체", None)]:
    for name, detector, norm in [("ORB", cv2.ORB_create(nfeatures=2000), cv2.NORM_HAMMING),
                                 ("SIFT", cv2.SIFT_create(), cv2.NORM_L2)]:
        r = feature_match(detector, norm, mask)
        print(f"  {region} {name:<4}: 특징점 {r['kp']:>9} | 좋은 매칭 {r['good']:4d} | inlier {r['inlier']:4d}"
              f" | 회전 {r['angle']:6.2f}도, 배율 {r['scale']:.3f} | 오차 H {r['error']:5.2f} / 닮음 {r['error_sim']:4.2f}"
              f" | {r['ms']:4.0f} ms")
        if region == "로고 영역만" and name == "SIFT":
            vis = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
            cv2.polylines(vis, [r["found"].astype(np.int32)], True, (0, 255, 0), 3)
            cv2.imwrite(str(OUT / "02_4_feature_found.jpg"), vis)
