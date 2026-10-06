"""16-4 프로젝트: 산업 Vision Hybrid Pipeline

Camera → OpenCV 품질 검사 → YOLO → Rule(치수) → SAM(결함 영역) → OCR(로트 번호) → VLM 상황 판단(필요할 때만) → 리포트

생산 라인 시뮬레이션: 부품 사진 아래쪽에 로트 번호 태그("LOT A123" 같은 가상의 번호)가 붙어 있다.
    정상 / 불량 6종(02장) / 처음 보는 불량(얼룩, 14-2) / 초점이 크게 나간 사진(재촬영 대상)
Routing (14장):
    - 품질 검사에서 떨어지면 재촬영 요청으로 끝 (뒤 단계를 부르지 않는다)
    - YOLO 불량 박스 신뢰도 0.6 이상 또는 Rule이 치수 불량을 확실히 찾으면 불량 확정
    - 애매(신뢰도 0.25~0.6 박스, Rule 여유 0.2 미만)하면 VLM이 판단
    - 확실한 정상 중 10%는 감사로 VLM에게도 보인다
    - 로트 번호가 형식(LOT + 영문 1자 + 숫자 3자)에 맞지 않으면 VLM에게 태그를 다시 읽게 한다

실행 (저장소 루트에서, 02-6·04-4 실습을 마치고 VLM 서버를 띄운 뒤):
    python ch16_projects/p4_hybrid_pipeline/pipeline.py --vlm-url http://localhost:8000/v1
    python ch16_projects/p4_hybrid_pipeline/pipeline.py --vlm-url http://localhost:8000/v1 --all-vlm   # 비교: 모든 사진을 VLM에게
"""
import argparse
import contextlib
import csv
import importlib.util
import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from rapidocr import RapidOCR
from transformers import SamModel, SamProcessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "ch10_vlm"))
import make_parts  # noqa: E402
from vlm import ask  # noqa: E402

LINE = ROOT / "data" / "parts_line"
OUT = ROOT / "outputs" / "ch16" / "p4"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--vlm-url", default="http://localhost:8000/v1")
ap.add_argument("--all-vlm", action="store_true", help="비교용: 모든 사진을 VLM이 판정")
args = ap.parse_args()
LOT = re.compile(r"^LOT [A-Z][0-9]{3}$")


# 0. 생산 라인 사진 만들기 — 부품 + 로트 번호 태그. 정답(불량 여부, 로트 번호, 재촬영 대상)을 함께 저장
def make_line(n=150, seed=2050):
    rng = np.random.default_rng(seed)
    LINE.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        kind = ["ok", "ok", "defect", "stain", "blurry"][int(rng.choice(5, p=[0.3, 0.2, 0.3, 0.1, 0.1]))]
        label = make_parts.DEFECTS[i % 6] if kind == "defect" else ("ok" if kind in ("ok", "blurry") else "stain")
        img, mask, _ = make_parts.make_image(rng, "ok" if label == "stain" else label, make_parts.LIGHTS[i % 3])
        if label == "stain":                                                  # 14-2와 같은 옅은 얼룩
            ys, xs = np.where(mask > 0)
            k = rng.integers(len(xs))
            spot = np.zeros(mask.shape, np.float32)
            cv2.ellipse(spot, (int(xs[k]), int(ys[k])), (24, 14), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
            img = np.clip(img * (1 - cv2.GaussianBlur(spot, (0, 0), 3) * (mask > 0) * 0.3), 0, 255).astype(np.uint8)
        lot = f"LOT {chr(65 + int(rng.integers(26)))}{int(rng.integers(1000)):03d}"
        cv2.rectangle(img, (230, 405), (410, 455), 235, -1)                   # 컨베이어 위의 흰 태그
        cv2.putText(img, lot, (242, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 20, 2, cv2.LINE_AA)
        if kind == "blurry":
            img = cv2.GaussianBlur(img, (0, 0), 6)                            # 초점이 크게 나감
        name = f"{i:03d}.png"
        cv2.imwrite(str(LINE / name), img)
        rows.append({"file": name, "label": label, "lot": lot, "blurry": kind == "blurry"})
    (LINE / "truth.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if not (LINE / "truth.json").exists():
    make_line()
truth = json.loads((LINE / "truth.json").read_text(encoding="utf-8"))


# 1. 단계별 도구 — 앞 장의 것을 그대로 불러온다
def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


rule = load_script(ROOT / "ch02_rule_based" / "02_6_rule_inspection.py")
yolo = load_script(ROOT / "ch04_detection" / "04_5_evaluate.py").model      # 04-4에서 학습한 부품·구멍·깨짐·균열 검출기
sam_proc, sam = SamProcessor.from_pretrained("facebook/sam-vit-base"), SamModel.from_pretrained("facebook/sam-vit-base").cuda().eval()
ocr = RapidOCR()
T = defaultdict(float)                                                        # 단계별 누적 시간
CALLS = Counter()


def timed(name):
    def wrap(fn):
        def inner(*a, **k):
            t = time.perf_counter()
            r = fn(*a, **k)
            T[name] += time.perf_counter() - t
            CALLS[name] += 1
            return r
        return inner
    return wrap


@timed("품질 검사")
def quality(gray):
    """초점(라플라시안 분산)과 밝기. 이 라인의 선명한 사진은 560 안팎, 크게 흐린 사진은 1 미만이라 50을 기준으로"""
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {"sharpness": round(sharp, 1), "brightness": round(float(gray.mean()), 1), "ok": sharp >= 50 and 20 <= gray.mean() <= 200}


@timed("YOLO")
def detect(path):
    """깨짐·균열 박스. 관심 영역(ROI): 신뢰도가 가장 높은 부품 박스 안에 중심이 있는 것만 쓴다
    (ROI가 없으면 태그의 글자 획을 균열로, 태그를 또 하나의 부품으로 본다 — 04-4의 학습 이미지에는 태그가 없었다)"""
    r = yolo(str(path), conf=0.25, verbose=False)[0]
    boxes = list(zip([r.names[int(c)] for c in r.boxes.cls.tolist()], r.boxes.conf.tolist(), r.boxes.xyxy.tolist()))
    parts = [b for n, s, b in sorted(boxes, key=lambda x: -x[1]) if n == "part"]
    if not parts:
        return []
    x1, y1, x2, y2 = parts[0]
    inside = lambda b: x1 <= (b[0] + b[2]) / 2 <= x2 and y1 <= (b[1] + b[3]) / 2 <= y2  # noqa: E731
    return [(n, s, b) for n, s, b in boxes if n in ("chip", "crack") and inside(b)]


@timed("Rule")
def measure(gray):
    f = rule.measure(gray, True)
    if f is None:
        return {"fails": ["부품 없음"], "margin": 1.0}
    fails = [x for x in rule.judge(f, rule.spec, True) if x in ("크기", "구멍 수", "각도", "위치", "꼭짓점")]
    margin = min(abs(0.06 - abs(f["area"] / rule.spec["area"] - 1)) / 0.06, abs(8 - abs(f["angle"])) / 8, abs(40 - f["offset"]) / 40)
    return {"fails": fails, "margin": round(margin, 2)}


@timed("SAM")
def region(path, box):
    image = Image.open(path).convert("RGB")
    inp = sam_proc(image, input_boxes=[[box]], return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = sam(**inp, multimask_output=False)
    m = sam_proc.image_processor.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0][0, 0]
    return int(m.sum())


def norm_lot(text):
    """'L0T B712', 'B712' 같은 답을 'LOT B712' 형식으로 맞춘다"""
    t = re.sub(r"\s+", " ", str(text).strip().upper().replace("L0T", "LOT"))
    return f"LOT {t}" if re.fullmatch(r"[A-Z][0-9]{3}", t) else t


@timed("OCR")
def read_lot(gray):
    """태그 자리를 알고 있으므로 글자 영역 찾기(det)·방향(cls) 없이 인식(rec)만 — 수십 배 빠르다"""
    r = ocr(gray[400:460, 220:420], use_det=False, use_cls=False)
    return norm_lot(" ".join(r.txts or []))


@timed("VLM")
def vlm_judge(path, context):
    q = ("This is an inspection photo of a metal bracket on a conveyor, with a white lot-number tag below it. "
         "A good bracket is a rounded rectangle with exactly two round holes, no broken corners, no cracks and no stains. "
         f"{context} Answer in JSON: {{\"defect\": true or false, \"lot\": \"the lot number on the tag\", \"reason\": \"one short sentence\"}}")
    text = ask(q, [Image.open(path)], max_tokens=120, base_url=args.vlm_url)["text"]
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


# 2. 파이프라인 — 사진 한 장 → 리포트 한 줄
rng = np.random.default_rng(0)


def inspect(row):
    path = LINE / row["file"]
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    rep = {"file": row["file"]}
    q = quality(gray)
    if not q["ok"]:
        return {**rep, "decision": "재촬영", "why": f"초점 {q['sharpness']}"}
    if args.all_vlm:
        v = vlm_judge(path, "")
        return {**rep, "decision": "불량" if v.get("defect") else "합격", "lot": norm_lot(v.get("lot", "")), "why": v.get("reason", ""), "by": "VLM"}
    boxes, geo, lot = detect(path), measure(gray), read_lot(gray)
    sure_bad = [b for b in boxes if b[1] >= 0.6]
    unsure = [b for b in boxes if b[1] < 0.6]
    rep["lot"] = lot
    if sure_bad or (geo["fails"] and geo["margin"] >= 0.2):
        rep.update(decision="불량", by="YOLO+Rule", why=", ".join([f"{n} {s:.2f}" for n, s, _ in sure_bad] + geo["fails"]))
        if sure_bad:
            rep["defect_area_px"] = region(path, sure_bad[0][2])               # 불량 영역을 SAM으로 정밀하게
    elif unsure or geo["margin"] < 0.2 or rng.random() < 0.1:                # 애매하거나 감사 표본이면 VLM
        why = "uncertain detector boxes" if unsure else ("close to a size/position limit" if geo["margin"] < 0.2 else "random audit")
        v = vlm_judge(path, f"An automatic inspector found this case {why}.")
        rep.update(decision="불량" if v.get("defect") else "합격", by=f"VLM ({why})", why=v.get("reason", ""))
    else:
        rep.update(decision="합격", by="YOLO+Rule", why="")
    if not LOT.match(lot):                                                    # 로트 번호가 형식에 안 맞으면 VLM에게 다시 읽게
        v = vlm_judge(path, "Read the lot number on the tag carefully.")
        rep["lot"], rep["lot_by"] = norm_lot(v.get("lot", "")), "VLM"
    return rep


t0 = time.perf_counter()
reports = [inspect(r) for r in truth]
sec = time.perf_counter() - t0
(OUT / ("report_all_vlm.json" if args.all_vlm else "report.json")).write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")

# 3. 정답과 맞춰 보기
ok_shot = [(r, t) for r, t in zip(reports, truth) if not t["blurry"]]
blur_hit = np.mean([r["decision"] == "재촬영" for r, t in zip(reports, truth) if t["blurry"]])
false_reshoot = sum(r["decision"] == "재촬영" for r, t in ok_shot)
judged = [(r, t) for r, t in ok_shot if r["decision"] != "재촬영"]
acc = np.mean([(r["decision"] == "불량") == (t["label"] != "ok") for r, t in judged])
per = defaultdict(list)
for r, t in judged:
    per[t["label"]].append((r["decision"] == "불량") == (t["label"] != "ok"))
lot_acc = np.mean([r.get("lot") == t["lot"] for r, t in judged])
print(f"{'모든 사진을 VLM이' if args.all_vlm else 'Hybrid Pipeline'}: 사진 {len(truth)}장 {sec:.0f}초 ({sec / len(truth) * 1000:.0f}ms/장)")
print(f"  재촬영: 흐린 사진 {sum(t['blurry'] for t in truth)}장 중 {blur_hit:.0%} 걸러냄, 멀쩡한 사진을 재촬영으로 {false_reshoot}장")
print(f"  판정 정확도 {acc:.1%} ({len(judged)}장) | " + " ".join(f"{k} {np.mean(v):.0%}({len(v)})" for k, v in sorted(per.items())))
print(f"  로트 번호 정확히 읽음 {lot_acc:.1%}" + ("" if args.all_vlm else f" (VLM이 다시 읽은 것 {sum('lot_by' in r for r, _ in judged)}장)"))
print("  단계별 호출(시간): " + " | ".join(f"{k} {CALLS[k]}번({T[k]:.1f}초)" for k in ["품질 검사", "YOLO", "Rule", "SAM", "OCR", "VLM"] if CALLS[k]))
by = Counter(r.get("by", "-").split(" (")[0] for r, _ in judged)
print(f"  판정한 단계: {dict(by)}")
for r, t in [(r, t) for r, t in judged if (r["decision"] == "불량") != (t["label"] != "ok")][:6]:
    print(f"  틀림: {r['file']} 정답 {t['label']} → {r['decision']} ({r.get('by', '')}) {r.get('why', '')[:80]}")
