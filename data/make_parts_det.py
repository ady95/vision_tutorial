"""04장 실습용 합성 부품 검출 데이터 만들기 (YOLO 형식)

02장의 부품을 한 장에 1~3개씩 배치하고, 정답 박스를 4종류로 자동 생성한다.
    0 part(부품)  1 hole(구멍)  2 chip(깨진 자리)  3 crack(균열)
구멍 누락(missing_hole)은 "없는 것"이라 박스를 그릴 수 없다 — 구멍 박스가 1개뿐인 부품으로만 나타난다.

실행:
    python data/make_parts_det.py      # data/parts_det/ 에 train 600, val 150, test 150장 + data.yaml
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from make_parts import H, HOLE_DX, HOLE_R, LIGHTS, PART_H, PART_W, W, draw_part, rounded_rect, to_image

NAMES = ["part", "hole", "chip", "crack"]
SLOTS = {1: [(320, 240)], 2: [(170, 240), (470, 240)], 3: [(165, 150), (475, 150), (320, 345)]}
SCALE = {1: (0.7, 1.0), 2: (0.55, 0.75), 3: (0.45, 0.58)}


def bbox(mask):
    ys, xs = np.where(mask > 0)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None


def make_scene(rng):
    k = int(rng.choice([1, 2, 3]))
    light = LIGHTS[int(rng.integers(3))]
    full = np.zeros((H, W), np.uint8)
    boxes = []
    for sx, sy in SLOTS[k]:
        scale = rng.uniform(*SCALE[k])
        angle = rng.uniform(-25, 25)
        cx, cy = sx + rng.uniform(-20, 20) * scale, sy + rng.uniform(-15, 15) * scale
        defect = str(rng.choice(["ok", "chip", "crack", "missing_hole"], p=[0.4, 0.2, 0.2, 0.2]))
        ok, bad = np.zeros_like(full), np.zeros_like(full)
        draw_part(ok, cx, cy, angle, scale, "ok")
        draw_part(bad, cx, cy, angle, scale, defect)
        full |= bad
        solid = np.zeros_like(full)                                  # 구멍까지 메운 부품 외곽
        cv2.fillPoly(solid, [to_image(rounded_rect(PART_W, PART_H, 20), cx, cy, angle, scale).round().astype(np.int32)], 255)
        broken_off = ok & ~bad                                       # 깨지거나 금이 가서 없어진 자리
        boxes.append((0, bbox(solid & ~broken_off)))
        holes = [(-HOLE_DX, 0)] if defect == "missing_hole" else [(-HOLE_DX, 0), (HOLE_DX, 0)]
        r = HOLE_R * scale
        for hx, hy in to_image(holes, cx, cy, angle, scale):
            boxes.append((1, (hx - r, hy - r, hx + r, hy + r)))
        if defect in ("chip", "crack"):
            boxes.append((NAMES.index(defect), bbox(broken_off)))
    img = 50 + rng.normal(0, 5, (H, W))
    img[full > 0] = 200 + rng.normal(0, 6, int((full > 0).sum()))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    if light == "gradient":
        img *= 0.08 + 1.1 * (np.arange(W) / W)[None, :] ** 2
    elif light == "dim":
        img *= 0.45
    img += rng.normal(0, 4, (H, W))
    return np.clip(img, 0, 255).astype(np.uint8), boxes, light


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "parts_det"))
    ap.add_argument("--seed", type=int, default=2028)
    args = ap.parse_args()
    out = Path(args.out)
    stats = {}
    for split, n, seed in [("train", 600, args.seed), ("val", 150, args.seed + 1), ("test", 150, args.seed + 2)]:
        rng = np.random.default_rng(seed)
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        counts = [0] * len(NAMES)
        for i in range(n):
            img, boxes, light = make_scene(rng)
            name = f"{i:04d}_{light}"
            cv2.imwrite(str(out / "images" / split / f"{name}.png"), img)
            lines = []
            for c, (x1, y1, x2, y2) in boxes:                       # YOLO 형식: 클래스 중심x 중심y 너비 높이 (0~1)
                lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                counts[c] += 1
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
        stats[split] = counts
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        + "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    for split, counts in stats.items():
        print(f"{split:<5} " + " | ".join(f"{n} {c}" for n, c in zip(NAMES, counts)))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
