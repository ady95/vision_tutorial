"""06-2. DETR: Transformer로 Detection하기

1) Bipartite Matching — 예측과 정답을 일대일로 짝짓기 (헝가리안 알고리즘)
2) DETR의 출력 — Query 100개, "물체 없음" 클래스, NMS 없는 결과
3) COCO val2017 500장에서 DETR · RT-DETR · YOLO26을 같은 방식(COCO mAP)으로 비교

실행 (저장소 루트에서):
    python data/download_coco_val.py
    python ch06_vit/06_2_detr.py
"""
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from transformers import AutoImageProcessor, DetrForObjectDetection
from ultralytics import RTDETR, YOLO
from ultralytics.data.converter import coco80_to_coco91_class

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Bipartite Matching — 정답 3개와 예측 5개를 일대일로 짝짓는다
gt = np.array([[10, 10, 50, 50], [60, 10, 100, 60], [20, 70, 70, 120]], float)
pred = np.array([[12, 8, 52, 49], [58, 12, 98, 62], [61, 9, 101, 58], [22, 72, 69, 118], [150, 150, 190, 190]], float)


def iou(a, b):
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clip(0).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter)


cost = 1 - iou(gt, pred)                                       # 겹칠수록 비용이 작다 (DETR은 클래스 확률·L1 거리도 더한다)
rows, cols = linear_sum_assignment(cost)                       # 비용 합이 최소인 일대일 짝
print("비용 행렬 (행: 정답 3개, 열: 예측 5개)\n", cost.round(2))
for r, c in zip(rows, cols):
    print(f"  정답 {r} ↔ 예측 {c} (IoU {1 - cost[r, c]:.2f})")
print(f"  짝이 없는 예측 {sorted(set(range(len(pred))) - set(cols))} → '물체 없음'으로 학습")

# 2. DETR의 출력 — Query 100개가 각자 하나씩 답한다
proc = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval().to(device)
img = Image.open(ROOT / "data" / "images" / "bus.jpg").convert("RGB")
with torch.no_grad():
    out = detr(**proc(images=img, return_tensors="pt").to(device))
prob = out.logits.softmax(-1)[0]                                # 마지막 칸이 "물체 없음"
print(f"\n[DETR] 파라미터 {sum(p.numel() for p in detr.parameters()):,} | Query {detr.config.num_queries}개")
print(f"  logits {tuple(out.logits.shape)} (Query 100 x 클래스 91+물체 없음 1) | pred_boxes {tuple(out.pred_boxes.shape)}")
print(f"  '물체 없음' 확률이 0.9를 넘는 Query: {int((prob[:, -1] > 0.9).sum())}개")
res = proc.post_process_object_detection(out, threshold=0.7, target_sizes=[img.size[::-1]])[0]
print(f"  신뢰도 0.7 이상: {len(res['scores'])}개 (NMS 없이 이것이 최종 결과)")
for s, l, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
    print(f"    {detr.config.id2label[l]:<11} {s:.2f} [{', '.join(f'{v:.0f}' for v in b)}]")

# 3. COCO val2017 500장 — 세 Detector를 같은 채점기(pycocotools)로
with contextlib.redirect_stdout(io.StringIO()):                # pycocotools의 진행 메시지는 숨긴다
    coco = COCO(str(COCO_DIR / "instances.json"))
images = coco.loadImgs(coco.getImgIds())
TO_COCO = coco80_to_coco91_class()                              # YOLO 클래스 번호(0~79) → COCO 번호(1~90)


def run_detr(path):
    im = Image.open(path).convert("RGB")
    with torch.no_grad():
        o = detr(**proc(images=im, return_tensors="pt").to(device))
    r = proc.post_process_object_detection(o, threshold=0.0, target_sizes=[im.size[::-1]])[0]   # Query 100개 전부
    return r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist()


def ultralytics_runner(model):
    def run(path):
        r = model(path, conf=0.001, max_det=300, device=device, verbose=False)[0]               # mAP 측정의 표준 설정
        return r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), [TO_COCO[int(c)] for c in r.boxes.cls]
    return run


detectors = {"DETR (ResNet-50)": (run_detr, sum(p.numel() for p in detr.parameters()))}
for name, m in [("RT-DETR-L", RTDETR("rtdetr-l.pt")), ("YOLO26s", YOLO("yolo26s.pt"))]:
    detectors[name] = (ultralytics_runner(m), sum(p.numel() for p in m.model.parameters()))

print(f"\nCOCO val2017 {len(images)}장, 정답 박스 {len(coco.getAnnIds())}개")
print(f"{'모델':<18}{'파라미터':>12}{'1장(ms)':>9}{'AP':>7}{'AP50':>7}{'AP_S':>7}{'AP_M':>7}{'AP_L':>7}")
for name, (run, params) in detectors.items():
    for _ in range(5):
        run(COCO_DIR / "images" / images[0]["file_name"])        # 워밍업
    dets, t = [], time.perf_counter()
    for im in images:
        boxes, scores, labels = run(COCO_DIR / "images" / im["file_name"])
        dets += [{"image_id": im["id"], "category_id": int(l), "score": float(s),
                  "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]} for b, s, l in zip(boxes, scores, labels)]
    ms = (time.perf_counter() - t) / len(images) * 1000
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(dets), "bbox")
        ev.evaluate(), ev.accumulate(), ev.summarize()
    s = ev.stats                                                # [AP, AP50, AP75, AP_S, AP_M, AP_L, ...]
    print(f"{name:<18}{params:>12,}{ms:>9.1f}{s[0]:>7.3f}{s[1]:>7.3f}{s[3]:>7.3f}{s[4]:>7.3f}{s[5]:>7.3f}")
    (ROOT / "outputs" / "ch06").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "ch06" / f"06_2_{name.split()[0]}_dets.json").write_text(json.dumps(dets), encoding="utf-8")
