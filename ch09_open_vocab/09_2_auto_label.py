"""09-2. 라벨 초안 자동 생성(Auto-labeling) — 04-4의 라벨링 비용을 줄이는 방법

1) Grounding DINO base로 04-4의 합성 부품 이미지(train 600, val 150, test 150)에 박스를 그린다
   글: "metal plate. hole. chip. crack." → 클래스 part / hole / chip / crack
2) 자동 라벨을 정답 라벨과 비교 — 클래스별 정밀도·재현율 (Rule 적용 전후)
   Rule: 부품판(metal plate) 박스와 거의 같은 chip 박스는 부품판을 잘못 부른 것이므로 지운다
3) 자동 라벨로 YOLO26n을 학습해, 사람이 만든 라벨로 학습한 04-4 모델과 같은 시험셋(정답 라벨)으로 비교

실행 (저장소 루트에서, data/make_parts_det.py 실행 후, 04-4 모델이 있으면 함께 비교):
    python ch09_open_vocab/09_2_auto_label.py
"""
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "parts_det"
DST = ROOT / "data" / "parts_det_auto"
RUNS = ROOT / "outputs" / "ch09" / "runs"
device = "cuda" if torch.cuda.is_available() else "cpu"
NAMES = ["part", "hole", "chip", "crack"]
PROMPT = "metal plate. hole. chip. crack."
KEYWORDS = [("crack", 3), ("chip", 2), ("hole", 1), ("plate", 0)]         # 표현에 들어 있는 단어로 클래스를 정한다

proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval().to(device)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(preds, gts):
    """IoU 0.5 이상인 짝을 하나씩 (겹친 정도가 큰 순서로) → 맞힌 개수"""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True)
    used_p, used_g = set(), set()
    for v, i, j in pairs:
        if v >= 0.5 and i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    return len(used_g)


def auto_label(path):
    """→ (Rule 적용 전 [(클래스, 박스)], Rule 적용 후)"""
    img = Image.open(path).convert("RGB")
    inp = proc(images=img, text=PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    r = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.35, text_threshold=0.25,
                                                    target_sizes=[img.size[::-1]])[0]
    raw = []
    for label, box in zip(r["text_labels"], r["boxes"].tolist()):
        cls = next((c for word, c in KEYWORDS if word in label), None)
        if cls is not None:
            raw.append((cls, box))
    plates = [b for c, b in raw if c == 0]
    ruled = [(c, b) for c, b in raw if not (c == 2 and any(iou(b, p) > 0.8 for p in plates))]
    return raw, ruled, img.size


def read_gt(label_path, size):
    w, h = size
    out = []
    for line in label_path.read_text().splitlines():
        c, x, y, bw, bh = line.split()
        x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
        out.append((int(c), [x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2]))
    return out


# 1. 자동 라벨 만들기 — 이미지는 복사하고 라벨만 새로 쓴다 (YOLO 형식)
t0 = time.perf_counter()
quality = {"Rule 전": defaultdict(lambda: [0, 0, 0]), "Rule 후": defaultdict(lambda: [0, 0, 0])}   # 클래스 → [맞힘, 예측, 정답]
n_img = 0
for split in ["train", "val", "test"]:
    (DST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
    for img_path in sorted((SRC / "images" / split).glob("*.png")):
        raw, ruled, (w, h) = auto_label(img_path)
        shutil.copy(img_path, DST / "images" / split / img_path.name)
        (DST / "labels" / split / f"{img_path.stem}.txt").write_text("".join(
            f"{c} {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}\n" for c, b in ruled))
        n_img += 1
        if split == "test":                                             # 2. 시험셋에서 정답과 비교
            gt = read_gt(SRC / "labels" / split / f"{img_path.stem}.txt", (w, h))
            for key, pred in [("Rule 전", raw), ("Rule 후", ruled)]:
                for c in range(4):
                    p = [b for k, b in pred if k == c]
                    g = [b for k, b in gt if k == c]
                    q = quality[key][c]
                    q[0] += match(p, g)
                    q[1] += len(p)
                    q[2] += len(g)
print(f"자동 라벨 {n_img}장, {time.perf_counter() - t0:.0f}초 (Grounding DINO base, Box 0.35 / Text 0.25)")
print(f"\n시험 150장에서 자동 라벨 vs 정답 라벨 (IoU 0.5)")
for key, q in quality.items():
    print(f"[{key}] " + " | ".join(f"{NAMES[c]} 박스 {q[c][1]}개(정답 {q[c][2]}) 정밀도 {q[c][0] / max(1, q[c][1]):.1%} 재현율 {q[c][0] / q[c][2]:.1%}"
                                   for c in range(4)))

# 3. 자동 라벨로 학습 → 정답 라벨로 시험
(DST / "data.yaml").write_text(f"path: {DST.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                               + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)), encoding="utf-8")
yolo = YOLO("yolo26n.pt")
t = time.perf_counter()
yolo.train(data=str(DST / "data.yaml"), epochs=50, imgsz=640, batch=16, device=0 if device == "cuda" else "cpu", seed=0,
           deterministic=True, project=str(RUNS), name="auto", exist_ok=True, verbose=False, plots=False)
minutes = (time.perf_counter() - t) / 60
print(f"\n자동 라벨로 학습: {minutes:.1f}분 (04-4와 같은 설정: YOLO26n 미세조정, 50에폭)")
print(f"{'시험(정답 라벨 150장)':<26}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}   클래스별 AP50 (part / hole / chip / crack)")
candidates = [("자동 라벨로 학습", RUNS / "auto" / "weights" / "best.pt"),
              ("사람 라벨로 학습 (04-4)", ROOT / "outputs" / "ch04" / "runs" / "finetune" / "weights" / "best.pt")]
for name, w in candidates:
    if not w.exists():
        print(f"{name:<26}(가중치 없음: 04-4를 먼저 실행하면 비교됩니다)")
        continue
    m = YOLO(w).val(data=str(SRC / "data.yaml"), split="test", device=0 if device == "cuda" else "cpu", verbose=False, plots=False).box
    print(f"{name:<26}{m.mp:>7.3f}{m.mr:>7.3f}{m.map50:>8.3f}{m.map:>10.3f}   " + " / ".join(f"{v:.3f}" for v in m.ap50))
