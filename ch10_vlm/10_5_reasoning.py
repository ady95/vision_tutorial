"""10-5. 실습: 관계 이해와 Visual Reasoning

1) 관계 질문 채점 — COCO 정답 박스로 "A가 B의 왼쪽에 있나?", "A가 B보다 위에 있나?"를 자동으로 만들어
   thinking을 껐을 때와 켰을 때의 정확도·시간을 비교한다
   (사진에 한 개씩만 있는 두 물체, 박스 중심이 화면 폭(높이)의 15% 넘게 떨어진 경우만, yes/no 절반씩)
2) 판단과 설명 — "이 차량이 정상적으로 주차되어 있는지 판단하고 이유를 설명하라" (thinking 끔/켬)

실행 (저장소 루트에서, vLLM 서버를 띄운 뒤):
    python ch10_vlm/10_5_reasoning.py
    python ch10_vlm/10_5_reasoning.py --images 내_차량_사진.jpg 다른_사진.jpg
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm import ask, model_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--n", type=int, default=120, help="관계 질문 수")
ap.add_argument("--images", nargs="*", default=[], help="주차 판단에 쓸 차량 사진")
args = ap.parse_args()
B = args.base_url
print(f"모델: {model_name(B)}")

# 1. 관계 질문 만들기
inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
names = {c["id"]: c["name"] for c in inst["categories"]}
by_img = defaultdict(list)
for a in inst["annotations"]:
    by_img[a["image_id"]].append(a)
rng = random.Random(0)
qs = []
for im in inst["images"]:
    anns = by_img[im["id"]]
    single = [a for a in anns if not a["iscrowd"] and sum(b["category_id"] == a["category_id"] for b in anns) == 1 and a["area"] > 32 ** 2]
    for i in range(len(single)):
        for j in range(i + 1, len(single)):
            a, b = single[i], single[j]
            ca = (a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2)
            cb = (b["bbox"][0] + b["bbox"][2] / 2, b["bbox"][1] + b["bbox"][3] / 2)
            na, nb = names[a["category_id"]], names[b["category_id"]]
            if abs(ca[0] - cb[0]) > 0.15 * im["width"]:
                truth = "yes" if ca[0] < cb[0] else "no"
                qs.append((im, f"Is the {na} to the left of the {nb} in this image? Answer yes or no.", truth, "좌우"))
            if abs(ca[1] - cb[1]) > 0.15 * im["height"]:
                truth = "yes" if ca[1] < cb[1] else "no"
                qs.append((im, f"Is the {na} above the {nb} in this image? Answer yes or no.", truth, "위아래"))
rng.shuffle(qs)
yes = [q for q in qs if q[2] == "yes"][:args.n // 2]
no = [q for q in qs if q[2] == "no"][:args.n // 2]
qs = yes + no
print(f"관계 질문 {len(qs)}개 (yes {len(yes)}, no {len(no)} / 좌우 {sum(q[3] == '좌우' for q in qs)}, 위아래 {sum(q[3] == '위아래' for q in qs)})")

print(f"{'':<12}{'정확도':>8}{'좌우':>8}{'위아래':>8}{'질문당 시간':>12}{'출력 토큰':>10}")
for think in [False, True]:
    ok, kind_ok, sec, toks = [], defaultdict(list), [], []
    for im, q, truth, kind in qs:
        r = ask(q, [Image.open(COCO_DIR / "images" / im["file_name"])], think=think, max_tokens=4096 if think else 8, base_url=B)
        pred = "yes" if r["text"].lower().startswith("yes") else "no"
        ok.append(pred == truth)
        kind_ok[kind].append(pred == truth)
        sec.append(r["seconds"])
        toks.append(r["completion_tokens"])
    print(f"{'thinking 켬' if think else 'thinking 끔':<12}{np.mean(ok):>8.1%}{np.mean(kind_ok['좌우']):>8.1%}{np.mean(kind_ok['위아래']):>8.1%}"
          f"{np.mean(sec):>11.2f}초{np.mean(toks):>10.0f}")

# 2. 판단과 설명
PROMPT = "이 차량이 정상적으로 주차되어 있는지 판단하고 이유를 두세 문장으로 설명하라. 번호판 글자는 쓰지 마라."
for path in args.images:
    img = Image.open(path)
    for think in [False, True]:
        r = ask(PROMPT, [img], think=think, max_tokens=4096, base_url=B, max_side=1280)
        print(f"\n[{Path(path).name}] thinking {'켬' if think else '끔'} ({r['seconds']:.1f}초, 생각 {len(r['thinking'])}자, 답 {r['completion_tokens']} 토큰)\n  → {r['text']}")
