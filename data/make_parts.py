"""02장 실습용 합성 부품 이미지 만들기

컨베이어 위의 금속 브래킷(구멍 2개)을 그린다. 정상과 불량 6종을, 조명 3종으로 만든다.
정답 마스크와 라벨(labels.csv)을 함께 저장하므로 결과를 숫자로 채점할 수 있다.

    불량: missing_hole(구멍 누락) chip(모서리 깨짐) crack(균열) rotated(회전) offset(위치 이탈) small(크기 불량)
    조명: normal(균일) gradient(왼쪽이 어두움) dim(전체가 어두움)

실행:
    python data/make_parts.py            # 조명별 60장(정상 30 + 불량 30)
    python data/make_parts.py --n 300    # 03장 분류 실습용으로 더 많이
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 480
PART_W, PART_H, CORNER, HOLE_R, HOLE_DX = 260, 150, 20, 20, 75
DEFECTS = ["missing_hole", "chip", "crack", "rotated", "offset", "small"]
LIGHTS = ["normal", "gradient", "dim"]


def rounded_rect(w, h, r, n=8):
    """중심이 원점인 둥근 사각형의 꼭짓점"""
    pts = []
    for cx, cy, a0 in [(w / 2 - r, -h / 2 + r, -90), (w / 2 - r, h / 2 - r, 0),
                       (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180)]:
        for a in np.linspace(a0, a0 + 90, n):
            pts.append((cx + r * np.cos(np.radians(a)), cy + r * np.sin(np.radians(a))))
    return np.array(pts)


def to_image(pts, cx, cy, angle, scale):
    """부품 좌표 → 이미지 좌표 (회전·확대·이동)"""
    t = np.radians(angle)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (np.asarray(pts, float) * scale) @ rot.T + (cx, cy)


def draw_part(mask, cx, cy, angle, scale, defect):
    poly = to_image(rounded_rect(PART_W, PART_H, CORNER), cx, cy, angle, scale)
    cv2.fillPoly(mask, [poly.round().astype(np.int32)], 255)
    holes = [(-HOLE_DX, 0), (HOLE_DX, 0)]
    if defect == "missing_hole":
        holes = holes[:1]
    for hx, hy in to_image(holes, cx, cy, angle, scale):
        cv2.circle(mask, (round(hx), round(hy)), round(HOLE_R * scale), 0, -1)
    if defect == "chip":       # 오른쪽 위 모서리를 삼각형으로 떼어 낸다
        tri = to_image([(PART_W / 2 - 70, -PART_H / 2 - 2), (PART_W / 2 + 2, -PART_H / 2 - 2),
                        (PART_W / 2 + 2, -PART_H / 2 + 55)], cx, cy, angle, scale)
        cv2.fillPoly(mask, [tri.round().astype(np.int32)], 0)
    if defect == "crack":      # 아래 가장자리에서 위로 70%까지 가는 금
        a, b = to_image([(10, PART_H / 2 + 2), (25, PART_H / 2 - PART_H * 0.7)], cx, cy, angle, scale)
        cv2.line(mask, tuple(a.round().astype(int)), tuple(b.round().astype(int)), 0, 4)


def make_image(rng, defect, light):
    angle, scale = rng.uniform(-4, 4), rng.uniform(0.98, 1.02)
    cx, cy = W / 2 + rng.uniform(-15, 15), H / 2 + rng.uniform(-15, 15)
    if defect == "rotated":
        angle = rng.choice([-1, 1]) * rng.uniform(15, 25)
    if defect == "offset":
        dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        d = rng.uniform(70, 100)
        cx, cy = cx + dx * d, cy + dy * d * 0.6
    if defect == "small":
        scale = rng.uniform(0.82, 0.86)

    mask = np.zeros((H, W), np.uint8)
    draw_part(mask, cx, cy, angle, scale, defect)

    img = 50 + rng.normal(0, 5, (H, W))                               # 컨베이어 벨트
    img[mask > 0] = 200 + rng.normal(0, 6, int((mask > 0).sum()))      # 금속 부품
    img = cv2.GaussianBlur(img, (3, 3), 0)
    if light == "gradient":     # 오른쪽에서 비추는 조명: 부품 왼쪽이 오른쪽 배경보다 어두워진다
        img *= 0.08 + 1.1 * (np.arange(W) / W)[None, :] ** 2
    elif light == "dim":
        img *= 0.45
    img += rng.normal(0, 4, (H, W))
    return np.clip(img, 0, 255).astype(np.uint8), mask, dict(angle=angle, scale=scale, cx=cx, cy=cy)


def make_multi(rng, out):
    """02-3 개수 세기용: 작은 부품 5개 + 먼지 3개"""
    mask = np.zeros((H, W), np.uint8)
    for cx, cy in [(130, 110), (330, 120), (520, 140), (200, 330), (450, 350)]:
        draw_part(mask, cx, cy, rng.uniform(-30, 30), 0.45, "ok")
    for _ in range(3):
        cv2.circle(mask, (int(rng.uniform(20, W - 20)), int(rng.uniform(20, H - 20))), 2, 255, -1)
    img = 50 + rng.normal(0, 5, (H, W))
    img[mask > 0] = 200
    img = np.clip(cv2.GaussianBlur(img, (3, 3), 0) + rng.normal(0, 4, (H, W)), 0, 255).astype(np.uint8)
    save(out / "multi.png", img)


def save(path, img):
    if not cv2.imwrite(str(path), img):                            # 실패해도 예외가 나지 않으므로 직접 확인한다
        raise SystemExit(f"저장 실패: {path} — Windows라면 경로에 한글이 없는 폴더로 저장소를 옮겨 실행하세요 (00-2)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="조명별 이미지 수 (절반은 정상)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "parts"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    n_ok = args.n // 2
    labels = ["ok"] * n_ok + [DEFECTS[i % len(DEFECTS)] for i in range(args.n - n_ok)]
    rows = []
    for light in LIGHTS:
        (out / light / "masks").mkdir(parents=True, exist_ok=True)
        for i, label in enumerate(labels):
            img, mask, meta = make_image(rng, label, light)
            name = f"{i:03d}_{label}.png"
            save(out / light / name, img)
            save(out / light / "masks" / name, mask)
            rows.append(dict(light=light, file=f"{light}/{name}", label=label,
                             **{k: round(float(v), 2) for k, v in meta.items()}))
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    make_multi(rng, out)
    print(f"{len(rows)}장 생성 → {out} (조명 {len(LIGHTS)}종 x {args.n}장, 정상 {n_ok} / 불량 {args.n - n_ok})")


if __name__ == "__main__":
    main()
