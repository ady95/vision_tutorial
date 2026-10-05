"""10-3. 실습: Captioning과 Visual Question Answering

1) "이 이미지를 설명하라" — bus.jpg, coco_cats.jpg
2) "자동차는 몇 대인가?" — COCO val2017 500장에서 사람·자동차 수를 묻고 정답 박스 개수와 비교 (YOLO26s 개수와 함께)
3) 장면 속 글자 읽기 — 05-5 영상 프레임의 차량 글자

실행 (저장소 루트에서, vLLM 서버를 띄운 뒤):
    python ch10_vlm/10_3_caption_vqa.py
    python ch10_vlm/10_3_caption_vqa.py --base-url http://localhost:8001/v1
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm import ask, first_int, model_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
args = ap.parse_args()
B = args.base_url
print(f"모델: {model_name(B)}")

# 1. 설명하기
for name in ["bus.jpg", "coco_cats.jpg"]:
    img = Image.open(ROOT / "data" / "images" / name)
    for prompt in ["Describe this image in one sentence.", "이 이미지를 한국어 두 문장으로 설명하라."]:
        r = ask(prompt, [img], base_url=B)
        print(f"[{name}] {prompt}\n  → {r['text']}  ({r['seconds']:.2f}초, 입력 {r['prompt_tokens']} / 출력 {r['completion_tokens']} 토큰)")

# 2. 개수 세기 — 정답 박스 개수와 비교
inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
names = {c["id"]: c["name"] for c in inst["categories"]}
count, crowd = defaultdict(Counter), defaultdict(set)
for a in inst["annotations"]:
    if a["iscrowd"]:
        crowd[a["image_id"]].add(names[a["category_id"]])                     # 무리(crowd)로 한 번에 표시된 클래스
    else:
        count[a["image_id"]][names[a["category_id"]]] += 1
yolo = YOLO("yolo26s.pt")
BUCKETS = [(1, 1), (2, 3), (4, 6), (7, 99)]
for cls, plural in [("person", "people"), ("car", "cars")]:
    imgs = [im for im in inst["images"] if count[im["id"]][cls] > 0 and cls not in crowd[im["id"]]]   # 무리(crowd) 표시가 있으면 정답 개수가 모호해 뺀다
    res = defaultdict(list)                                                   # 방법 → [(정답, 답)]
    t = time.perf_counter()
    for im in imgs:
        image = Image.open(COCO_DIR / "images" / im["file_name"])
        gt = count[im["id"]][cls]
        ans = first_int(ask(f"How many {plural} are in this image? Answer with a number only.", [image], max_tokens=16, base_url=B)["text"])
        res["VLM"].append((gt, ans))
        yr = yolo(image, conf=0.25, classes=[0 if cls == "person" else 2], device="cpu", verbose=False)[0]   # GPU는 VLM 서버가 쓰므로 CPU
        res["YOLO26s"].append((gt, len(yr.boxes)))
    sec = time.perf_counter() - t
    print(f"\n'{plural}' 세기: 사진 {len(imgs)}장 ({sec:.0f}초, VLM과 YOLO 합계)")
    print(f"{'':<10}{'정확히 맞힘':>10}{'평균 오차':>9}" + "".join(f"{f'정답 {lo}' + ('' if lo == hi else f'~{hi}' if hi < 99 else '+'):>12}" for lo, hi in BUCKETS))
    for key, pairs in res.items():
        ok = [a == g for g, a in pairs]
        err = [abs((a if a is not None else 0) - g) for g, a in pairs]
        row = f"{key:<10}{np.mean(ok):>10.1%}{np.mean(err):>9.2f}"
        for lo, hi in BUCKETS:
            sel = [o for (g, _), o in zip(pairs, ok) if lo <= g <= hi]
            row += f"{f'{np.mean(sel):.0%} ({len(sel)}장)':>12}" if sel else f"{'-':>12}"
        print(row)
    many = [(g, a) for g, a in res["VLM"] if g >= 7]
    print(f"  정답 7개 이상에서 VLM의 답(정답→답): " + ", ".join(f"{g}→{a}" for g, a in many[:12]))

# 3. 장면 속 글자 — 05-5 영상 1,449번째 프레임
cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 1448)
frame = Image.fromarray(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
for prompt in ["Read all the text you can see on the vehicles, except license plates. List each with the vehicle it is on.",
               "What company's delivery van is visible in this image?"]:
    r = ask(prompt, [frame], max_tokens=300, base_url=B, max_side=1280)       # 1920x1080은 12GB GPU의 4B에서 메모리 부족
    print(f"\n[1449프레임] {prompt}\n  → {r['text']}  ({r['seconds']:.2f}초)")
