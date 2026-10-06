"""16-1 프로젝트 1단계 — 결함 영역 마스크가 붙은 학습 데이터 만들기

make_parts.py로 부품을 그리고, 같은 위치·각도·크기의 "정상 부품" 모양과 비교해 결함 영역(없어진 부분)을 마스크로 얻는다.
    결함 영역이 있는 불량: missing_hole(메워진 구멍), chip(떨어져 나간 모서리), crack(갈라진 금)
    결함 영역이 없는 불량: rotated, offset, small → 영역이 아니라 치수로 판정한다 (inspect.py)
YOLO 분할 형식(클래스 + 다각형)으로 저장한다.

실행 (저장소 루트에서):
    python ch16_projects/p1_defect_inspection/make_data.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
import make_parts  # noqa: E402

OUT = ROOT / "data" / "parts_seg"
CLASSES = ["missing_hole", "chip", "crack"]                                   # 영역으로 찾는 결함


def defect_mask(meta, defect):
    """같은 자리에 정상 부품을 그려 비교한다. 정상에는 있는데 불량에는 없는 화소 = 결함 영역
    (missing_hole은 반대로, 정상에는 없는데(구멍) 불량에는 있는(메워진) 화소)"""
    ok = np.zeros((make_parts.H, make_parts.W), np.uint8)
    bad = np.zeros_like(ok)
    make_parts.draw_part(ok, meta["cx"], meta["cy"], meta["angle"], meta["scale"], "ok")
    make_parts.draw_part(bad, meta["cx"], meta["cy"], meta["angle"], meta["scale"], defect)
    diff = cv2.absdiff(ok, bad)
    return cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # 가장자리 1화소 차이는 지운다


def polygons(mask):
    """마스크 → YOLO 분할 형식의 다각형 (0~1로 정규화한 x y 쌍)"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < 20 or len(c) < 3:
            continue
        out.append(" ".join(f"{x / make_parts.W:.5f} {y / make_parts.H:.5f}" for x, y in c[:, 0]))
    return out


def main():
    rng = np.random.default_rng(2040)                                         # 다른 장의 seed(2026~2032)와 겹치지 않게
    for split, n in [("train", 600), ("val", 150)]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "masks" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            light = make_parts.LIGHTS[i % 3]
            label = "ok" if i % 2 == 0 else make_parts.DEFECTS[(i // 2) % 6]  # 정상 절반, 불량 6종 고르게
            img, _, meta = make_parts.make_image(rng, label, light)
            name = f"{i:04d}_{light}_{label}"
            cv2.imwrite(str(OUT / "images" / split / f"{name}.png"), img)
            lines, full = [], np.zeros(img.shape, np.uint8)
            if label in CLASSES:
                m = defect_mask(meta, label)
                full = m
                lines = [f"{CLASSES.index(label)} {p}" for p in polygons(m)]
            cv2.imwrite(str(OUT / "masks" / split / f"{name}.png"), full)          # 영역 정확도 채점용 정답 마스크
            (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "data.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames: {CLASSES}\n", encoding="utf-8")
    print(f"→ {OUT}: train 600, val 150 (조명 3종, 정상 절반, 불량 6종), 영역 라벨 클래스 {CLASSES}")


if __name__ == "__main__":
    main()
