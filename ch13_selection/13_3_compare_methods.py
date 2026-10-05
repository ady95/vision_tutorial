"""13-3. 같은 문제, 여러 방법 — 정확도 · 속도 · 메모리를 한 표로

COCO val2017 500장에서 "X를 모두 찾아라" (07-4·08-4·09-2·10-4와 같은 1,985문항, 같은 채점: 박스 IoU 0.5)
- YOLO26n, YOLO26s: 정해진 80종만 아는 검출기. 사진마다 한 번 돌리고 물은 클래스만 남긴다
- YOLOE-26s: 이름을 글로 넣어 클래스를 정하는 Open-Vocabulary YOLO (09-2)
- Grounding DINO tiny: 질문마다 글로 찾는다 (09-2, Box Threshold 0.35)
- (선택) VLM: --vlm-url을 주면 10-4와 같은 질문으로 함께 잰다 (오래 걸린다)

실행 (저장소 루트에서):
    python ch13_selection/13_3_compare_methods.py
    python ch13_selection/13_3_compare_methods.py --vlm-url http://localhost:8000/v1
"""
import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import YOLO, YOLOE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ch10_vlm"))
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
ap = argparse.ArgumentParser()
ap.add_argument("--vlm-url", default="", help="VLM 서버 주소 (생략하면 VLM은 건너뛴다)")
args = ap.parse_args()

inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
names = {c["id"]: c["name"] for c in inst["categories"]}
gt = defaultdict(lambda: defaultdict(list))
for a in inst["annotations"]:
    if not a["iscrowd"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]][names[a["category_id"]]].append([x, y, x + w, y + h])
rng = random.Random(0)                                                        # 앞 장들과 같은 "없는 물체" 질문
QUESTIONS = []                                                                # (이미지, [물을 클래스], 없는 클래스)
for im in inst["images"]:
    present = list(gt[im["id"]])
    absent = rng.choice([n for n in names.values() if n not in present])
    QUESTIONS.append((im, present + [absent], absent))


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy)


def match(preds, gts):
    used_p, used_g = set(), set()
    for v, i, j in sorted(((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True):
        if v >= 0.5 and i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    return len(used_g)


# 방법마다 "사진 한 장과 클래스 목록 → {클래스: 박스 목록}"을 돌려주는 함수
def per_image(model):
    """사진마다 한 번 돌리고, 물은 클래스의 박스만 나눠 담는다 (YOLO, YOLOE)"""
    def run(image, classes):
        r = model(image, conf=0.25, verbose=False)[0]
        out = defaultdict(list)
        for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            out[r.names[int(c)]].append(b)
        return {cls: out[cls] for cls in classes}
    return run


def grounding_dino(mid="IDEA-Research/grounding-dino-tiny", threshold=0.35):
    proc, model = AutoProcessor.from_pretrained(mid), AutoModelForZeroShotObjectDetection.from_pretrained(mid).eval().cuda()

    def run(image, classes):                                                  # 질문(클래스)마다 한 번
        out = {}
        for cls in classes:
            inp = proc(images=image, text=f"{cls}.", return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model(**inp)
            r = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=threshold, text_threshold=0.25,
                                                            target_sizes=[image.size[::-1]])[0]
            out[cls] = r["boxes"].tolist()
        return out
    return run, sum(p.numel() for p in model.parameters())


def vlm(url):
    from vlm import ask

    def run(image, classes):
        out = {}
        for cls in classes:
            text = ask(f'Find every {cls} in this image. Output a JSON list like [{{"label": "{cls}", "bbox_2d": [x1, y1, x2, y2]}}]. '
                       f"If there is no {cls}, output [].", [image], max_tokens=1024, base_url=url)["text"]
            w, h = image.size
            out[cls] = [[v[0] / 1000 * w, v[1] / 1000 * h, v[2] / 1000 * w, v[3] / 1000 * h]
                        for v in ([float(x) for x in re.findall(r"-?\d+\.?\d*", m)] for m in re.findall(r'"bbox_2d"\s*:\s*\[(.*?)\]', text))
                        if len(v) == 4]
        return out
    return run


def methods():
    """하나씩 불러와 재고 버린다 (GPU 최대 메모리를 방법마다 따로 재기 위해)"""
    for name, label in [("yolo26n.pt", "YOLO26n"), ("yolo26s.pt", "YOLO26s")]:
        m = YOLO(name)
        yield f"{label} (80종 고정)", per_image(m), sum(p.numel() for p in m.model.parameters())
    m = YOLOE("yoloe-26s-seg.pt")
    m.set_classes(list(names.values()))                                       # 80종의 이름을 글로 넣어 클래스를 정한다
    yield "YOLOE-26s (이름으로 정함)", per_image(m), sum(p.numel() for p in m.model.parameters())
    run, n = grounding_dino()
    yield "G-DINO tiny @0.35", run, n
    if args.vlm_url:
        from vlm import model_name
        yield f"VLM {model_name(args.vlm_url).split('/')[-1]}", vlm(args.vlm_url), None


print(f"COCO val2017 {len(QUESTIONS)}장, 질문 {sum(len(c) for _, c, _ in QUESTIONS)}개 (있는 물체 + 사진마다 없는 물체 1개)")
print(f"{'방법':<26}{'파라미터':>12}{'재현율':>8}{'정밀도':>8}{'없는데 찾음':>10}{'장당(ms)':>10}{'장/초':>7}{'GPU 최대':>10}")
for name, run, params in methods():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run(Image.open(COCO_DIR / "images" / QUESTIONS[0][0]["file_name"]).convert("RGB"), ["person"])     # 워밍업
    s = defaultdict(int)
    t = time.perf_counter()
    for im, classes, absent in QUESTIONS:
        preds = run(Image.open(COCO_DIR / "images" / im["file_name"]).convert("RGB"), classes)
        for cls in classes:
            if cls == absent:
                s["없음"] += 1
                s["지어냄"] += bool(preds[cls])
            else:
                s["정답"] += len(gt[im["id"]][cls])
                s["예측"] += len(preds[cls])
                s["맞힘"] += match(preds[cls], gt[im["id"]][cls])
    ms = (time.perf_counter() - t) / len(QUESTIONS) * 1000
    mem = f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if params else "서버"
    print(f"{name:<26}{(f'{params / 1e6:.1f}M' if params else '-'):>12}{s['맞힘'] / s['정답']:>8.1%}{s['맞힘'] / max(1, s['예측']):>8.1%}"
          f"{s['지어냄'] / s['없음']:>10.1%}{ms:>10.1f}{1000 / ms:>7.1f}{mem:>10}")
