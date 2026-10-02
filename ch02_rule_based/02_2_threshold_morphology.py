"""02-2. Threshold, Blur, Morphology

합성 부품 이미지(data/make_parts.py)로 이진화 방법을 조명별로 비교하고,
Noise 제거와 Morphology의 효과를 정답 마스크와의 IoU로 잰다.

실행:
    python data/make_parts.py
    python ch02_rule_based/02_2_threshold_morphology.py
"""
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "data" / "parts"
OUT = ROOT / "outputs" / "ch02"
OUT.mkdir(parents=True, exist_ok=True)


def load(light, name):
    img = cv2.imread(str(PARTS / light / name), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(PARTS / light / "masks" / name), cv2.IMREAD_GRAYSCALE)
    return img, gt


def iou(pred, gt):
    p, g = pred > 0, gt > 0
    return (p & g).sum() / (p | g).sum()


def fixed(img):
    return cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]


def otsu(img):
    t, b = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def adaptive(img):
    return cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -5)


def background_corrected(img):
    """조명 보정: 부품보다 큰 커널의 Opening으로 배경 밝기를 추정하고 나눈 뒤 Otsu

    Opening은 최솟값 필터에서 시작하므로 Noise가 있으면 배경을 너무 낮게 잡는다 → 앞뒤로 흐리게 한다
    """
    smooth = cv2.GaussianBlur(img, (9, 9), 0)
    bg = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (171, 171)))
    bg = cv2.GaussianBlur(bg, (0, 0), 25)
    flat = cv2.divide(img, cv2.max(bg, 1), scale=64)
    return otsu(flat)


# 1. 이진화 방법 4가지 x 조명 3가지
methods = {"고정값 127": fixed, "Otsu": otsu, "Adaptive": adaptive, "조명 보정 + Otsu": background_corrected}
print(f"{'방법':<14}" + "".join(f"{light:>10}" for light in ["normal", "dim", "gradient"]) + "   시간(ms)")
for name, fn in methods.items():
    scores, ms = [], 0.0
    for light in ["normal", "dim", "gradient"]:
        img, gt = load(light, "000_ok.png")
        t = time.perf_counter()
        b = fn(img)
        ms += (time.perf_counter() - t) * 1000 / 3
        scores.append(iou(b, gt))
        cv2.imwrite(str(OUT / f"02_2_{light}_{fn.__name__}.png"), b)
    print(f"{name:<14}" + "".join(f"{s:>10.3f}" for s in scores) + f"   {ms:8.2f}")

img, _ = load("normal", "000_ok.png")
t_otsu, _ = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img_dim, _ = load("dim", "000_ok.png")
t_dim, _ = cv2.threshold(cv2.GaussianBlur(img_dim, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"\nOtsu가 고른 임계값: normal {t_otsu:.0f}, dim {t_dim:.0f}")

# 2. Noise 제거 — 소금·후추 Noise(2%)를 넣고 필터별로 비교
img, gt = load("normal", "000_ok.png")
rng = np.random.default_rng(0)
noisy = img.copy()
pos = rng.random(img.shape)
noisy[pos < 0.01] = 0
noisy[pos > 0.99] = 255
filters = {
    "필터 없음": noisy,
    "Gaussian 5x5": cv2.GaussianBlur(noisy, (5, 5), 0),
    "Median 5x5": cv2.medianBlur(noisy, 5),
    "Bilateral": cv2.bilateralFilter(noisy, 9, 75, 75),
}
print(f"\n{'필터':<14}{'IoU':>7}{'조각 수':>8}")
for name, f in filters.items():
    _, b = cv2.threshold(f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _ = cv2.connectedComponents(b)
    print(f"{name:<14}{iou(b, gt):>7.3f}{n - 1:>8}")

# 3. Morphology — Opening은 흰 점을, Closing은 검은 점을 지운다
_, b = cv2.threshold(noisy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k)
print(f"\n{'단계':<14}{'IoU':>7}{'조각 수':>8}")
for name, m in [("이진화 직후", b), ("Opening", opened), ("+ Closing", cleaned)]:
    print(f"{name:<14}{iou(m, gt):>7.3f}{cv2.connectedComponents(m)[0] - 1:>8}")

# 4. 주의: Closing은 진짜 불량(균열)까지 메운다
img, gt = load("normal", "032_crack.png")
b = otsu(img)
for size in [3, 5, 9]:
    closed = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = max(cv2.arcLength(c, True) for c in contours)
    print(f"균열 이미지 Closing {size}x{size}: 바깥 윤곽 둘레 {perim:7.1f}")
cv2.imwrite(str(OUT / "02_2_crack_closed9.png"),
            cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))))
