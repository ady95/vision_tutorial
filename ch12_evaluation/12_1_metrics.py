"""12-1. Task별 평가 지표 — 앞 장에서 다루지 않은 것을 직접 계산한다

1) Detection: mAP는 클래스 평균(macro)이다 — YOLO26s, COCO val2017 500장. 물체 수로 가중하면?
2) Segmentation: 박스 AP와 마스크 AP, 인스턴스마다 IoU와 Dice — YOLO26s-seg
3) Tracking: MOTA · IDF1 · HOTA가 서로 다른 것을 벌한다 — 정답을 아는 합성 장면에 트래커 네 종류를 흉내 낸다

실행 (저장소 루트에서):
    python ch12_evaluation/12_1_metrics.py
"""
import contextlib
import io
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.data.converter import coco80_to_coco91_class

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
TO_COCO = coco80_to_coco91_class()
with contextlib.redirect_stdout(io.StringIO()):
    coco = COCO(str(COCO_DIR / "instances.json"))
images = coco.loadImgs(coco.getImgIds())


def evaluate(dets, kind):
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(dets), kind)
        ev.evaluate(), ev.accumulate(), ev.summarize()
    return ev


# 1. Detection — mAP는 클래스마다 AP를 구해 평균한 값
model = YOLO("yolo26s.pt")
dets = []
for im in images:
    r = model(COCO_DIR / "images" / im["file_name"], conf=0.001, max_det=300, verbose=False)[0]
    dets += [{"image_id": im["id"], "category_id": TO_COCO[int(c)], "score": float(s), "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]}
             for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
ev = evaluate(dets, "bbox")
prec = ev.eval["precision"][:, :, :, 0, 2]                                    # [IoU 10단계, recall 101점, 클래스, 넓이 all, maxDets 100]
n_gt = Counter(a["category_id"] for a in coco.loadAnns(coco.getAnnIds()) if not a["iscrowd"])
per_class = {}
for k, cid in enumerate(ev.params.catIds):
    p = prec[:, :, k]
    if n_gt[cid] and (p > -1).any():
        per_class[cid] = float(np.mean(p[p > -1]))
w = np.array([n_gt[c] for c in per_class])
ap = np.array(list(per_class.values()))
print(f"[Detection] YOLO26s, COCO val2017 {len(images)}장, 정답이 있는 클래스 {len(per_class)}개")
print(f"  mAP (클래스 평균, COCO 공식) {ev.stats[0]:.3f} | 물체 수로 가중한 평균 {np.sum(ap * w) / w.sum():.3f}")
print(f"  클래스별 AP: 최고 {ap.max():.3f} · 최저 {ap.min():.3f} · 표준편차 {ap.std():.3f}")
order = sorted(per_class, key=per_class.get)
name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
print("  AP 낮은 5개: " + ", ".join(f"{name[c]} {per_class[c]:.2f}({n_gt[c]}개)" for c in order[:5]))
print("  AP 높은 5개: " + ", ".join(f"{name[c]} {per_class[c]:.2f}({n_gt[c]}개)" for c in order[-5:]))
few = [c for c in per_class if n_gt[c] <= 5]
print(f"  정답 5개 이하인 클래스 {len(few)}개 — 이 클래스들의 AP 평균 {np.mean([per_class[c] for c in few]):.3f} (mAP에서 각각 1/{len(per_class)}의 무게)")

# 2. Segmentation — 박스 AP vs 마스크 AP, 그리고 인스턴스마다 IoU와 Dice
seg = YOLO("yolo26s-seg.pt")
bdets, mdets, pairs = [], [], []
for im in images:
    h, w_ = im["height"], im["width"]
    r = seg(COCO_DIR / "images" / im["file_name"], conf=0.001, max_det=100, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        continue
    masks = r.masks.data.cpu().numpy() > 0.5                                  # 원본 크기 마스크 (retina_masks)
    for m, b, s, c in zip(masks, r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        bdets.append({"image_id": im["id"], "category_id": TO_COCO[int(c)], "score": float(s), "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]})
        rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))      # COCO 형식(RLE)으로
        rle["counts"] = rle["counts"].decode()
        mdets.append({"image_id": im["id"], "category_id": TO_COCO[int(c)], "score": float(s), "segmentation": rle})
    keep = [i for i, s in enumerate(r.boxes.conf.tolist()) if s >= 0.25]     # 인스턴스 비교는 실제로 쓰는 기준(0.25)으로
    for a in coco.loadAnns(coco.getAnnIds(imgIds=im["id"], iscrowd=False)):
        g = coco.annToMask(a).astype(bool)
        best = (0.0, 0.0)
        for i in keep:
            if TO_COCO[int(r.boxes.cls[i])] != a["category_id"]:
                continue
            inter = np.logical_and(g, masks[i]).sum()
            union = np.logical_or(g, masks[i]).sum()
            iou = inter / union
            if iou > best[0]:
                best = (iou, 2 * inter / (g.sum() + masks[i].sum()))
        pairs.append((a["area"], *best))
eb, em = evaluate(bdets, "bbox"), evaluate(mdets, "segm")
print(f"\n[Segmentation] YOLO26s-seg: 박스 AP {eb.stats[0]:.3f} / 마스크 AP {em.stats[0]:.3f} "
      f"(작은 물체: 박스 {eb.stats[3]:.3f} / 마스크 {em.stats[3]:.3f})")
hit = [(a, i, d) for a, i, d in pairs if i >= 0.5]
print(f"  정답 {len(pairs)}개 중 마스크 IoU 0.5 이상으로 찾은 것 {len(hit)}개 — 그 평균 IoU {np.mean([i for _, i, _ in hit]):.3f}, "
      f"평균 Dice {np.mean([d for _, _, d in hit]):.3f}")
for lo, hi, label in [(0, 32 ** 2, "작은 물체"), (32 ** 2, 96 ** 2, "중간"), (96 ** 2, 1e12, "큰 물체")]:
    sel = [(i, d) for a, i, d in hit if lo <= a < hi]
    print(f"  {label:<6} 평균 IoU {np.mean([i for i, _ in sel]):.3f} · Dice {np.mean([d for _, d in sel]):.3f} ({len(sel)}개)")


# 3. Tracking — 정답을 아는 합성 장면
def scene():
    """100프레임, 물체 3개. 0번과 1번은 50프레임 근처에서 엇갈린다"""
    gt = defaultdict(dict)                                                    # 프레임 → {물체 ID: 박스}
    for t in range(100):
        gt[t][0] = (10 + 6 * t, 100)
        gt[t][1] = (610 - 6 * t, 110)
        gt[t][2] = (300, 300 + t)
    return {t: {k: (x, y, x + 40, y + 40) for k, (x, y) in objs.items()} for t, objs in gt.items()}


GT = scene()
TRACKERS = {
    "A: 30프레임 놓침": {t: {k: b for k, b in o.items() if not (k == 0 and 30 <= t < 60)} for t, o in GT.items()},
    "B: 엇갈릴 때 ID 교환": {t: {({0: 1, 1: 0}.get(k, k) if t >= 50 else k): b for k, b in o.items()} for t, o in GT.items()},
    "C: ID가 20프레임마다 새로": {t: {(k if k != 2 else 10 + t // 20): b for k, b in o.items()} for t, o in GT.items()},
    "D: 가짜 물체 40프레임": {t: {**o, **({99: (500, 400, 540, 440)} if t < 40 else {})} for t, o in GT.items()},
}


def box_iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy)


def match(g, p, th=0.5):
    """한 프레임의 정답-예측 짝 (IoU가 큰 쪽부터, Hungarian)"""
    if not g or not p:
        return []
    gk, pk = list(g), list(p)
    cost = np.array([[-box_iou(g[a], p[b]) for b in pk] for a in gk])
    rows, cols = linear_sum_assignment(cost)
    return [(gk[r], pk[c]) for r, c in zip(rows, cols) if -cost[r, c] >= th]


def track_metrics(gt, pred):
    tp = fn = fp = idsw = 0
    last = {}
    pairs = Counter()                                                         # (정답 ID, 예측 ID) → 짝지어진 프레임 수
    overlap = Counter()                                                       # (정답 ID, 예측 ID) → IoU 0.5 이상으로 겹친 프레임 수
    n_g, n_p = Counter(), Counter()
    tps = []
    for t in gt:
        m = match(gt[t], pred.get(t, {}))
        tp, fn, fp = tp + len(m), fn + len(gt[t]) - len(m), fp + len(pred.get(t, {})) - len(m)
        for g, p in m:
            idsw += g in last and last[g] != p                                # 같은 물체에 붙은 예측 ID가 바뀜
            last[g] = p
            pairs[(g, p)] += 1
            tps.append((g, p))
        for g, gb in gt[t].items():
            for p, pb in pred.get(t, {}).items():
                overlap[(g, p)] += box_iou(gb, pb) >= 0.5
        n_g.update(list(gt[t]))
        n_p.update(list(pred.get(t, {})))
    total = sum(n_g.values())
    mota = 1 - (fn + fp + idsw) / total
    # IDF1: 정답 ID와 예측 ID를 1:1로 묶어(영상 전체에서 한 번) 겹친 프레임 수를 최대로
    gk, pk = list(n_g), list(n_p)
    cost = np.array([[-overlap[(g, p)] for p in pk] for g in gk])
    r, c = linear_sum_assignment(cost)
    idtp = -cost[r, c].sum()
    idf1 = 2 * idtp / (total + sum(n_p.values()))
    # HOTA (IoU 기준 0.5 하나): 검출 정확도 DetA와 연결 정확도 AssA의 기하평균
    det_a = tp / (tp + fn + fp)
    ass_a = np.mean([pairs[(g, p)] / (n_g[g] + n_p[p] - pairs[(g, p)]) for g, p in tps])
    return mota, idf1, np.sqrt(det_a * ass_a), det_a, ass_a, idsw


print(f"\n[Tracking] 합성 장면 100프레임, 물체 3개 (정답 박스 {sum(len(o) for o in GT.values())}개)")
print(f"{'트래커':<22}{'MOTA':>7}{'IDF1':>7}{'HOTA':>7}{'DetA':>7}{'AssA':>7}{'ID 전환':>8}")
for k, pred in TRACKERS.items():
    mota, idf1, hota, det_a, ass_a, idsw = track_metrics(GT, pred)
    print(f"{k:<22}{mota:>7.3f}{idf1:>7.3f}{hota:>7.3f}{det_a:>7.3f}{ass_a:>7.3f}{idsw:>8}")


def with_motmetrics(gt, pred):
    """같은 장면을 표준 구현(motmetrics)으로. 거리 = 1 - IoU, IoU 0.5 미만은 짝이 될 수 없음(nan)"""
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=True)
    for t in gt:
        p = pred.get(t, {})
        dist = [[1 - box_iou(a, b) if box_iou(a, b) >= 0.5 else np.nan for b in p.values()] for a in gt[t].values()]
        acc.update(list(gt[t]), list(p), dist)       # motmetrics의 iou_matrix는 NumPy 2에서 동작하지 않아 직접 계산
    return mm.metrics.create().compute(acc, metrics=["mota", "idf1", "num_switches"]).iloc[0]


try:
    for k, pred in TRACKERS.items():
        s = with_motmetrics(GT, pred)
        print(f"  motmetrics {k[:1]}: MOTA {s['mota']:.3f} · IDF1 {s['idf1']:.3f} · ID 전환 {int(s['num_switches'])}")
except ImportError:
    print("  (pip install motmetrics로 표준 구현과 대조할 수 있다)")
