"""16-2 프로젝트: 차량 번호판 인식 시스템

Vehicle Detection(YOLO26s) + Tracking(ByteTrack) → Plate Detection(Rule 04-6 / Grounding DINO) → 정면화(02-5)
→ OCR(RapidOCR: PaddleOCR 모델) → 형식 규칙 → 여러 프레임 다수결 → (애매하면) VLM 검증

단계 (--step):
    collect     05-5의 교통 영상에서 차량을 추적하며, 프레임마다 번호판을 찾고 읽는다 → outputs/ch16/p2/frames.json
    time        단계별 시간만 따로 잰다 (영상의 차량 200개로)
    candidates  차량마다 가장 가까운 프레임의 번호판 사진과 읽기 후보를 모은다 (정답을 만들 때만. 번호판이 보이므로
                data/plate_gt/ 에만 저장하고 저장소에 올리지 않는다)
    eval        정답(data/plate_gt/gt.json)과 비교: 번호판 찾기, 한 프레임 OCR, 다수결, VLM 검증, 오류 유형
출력에는 번호판 글자를 찍지 않는다. 정확도와 개수만 출력한다.

실행 (저장소 루트에서, 04-6 실습 후):
    python ch16_projects/p2_plate_recognition/plate_pipeline.py --step collect
    python ch16_projects/p2_plate_recognition/plate_pipeline.py --step eval --vlm-url http://localhost:8000/v1
"""
import argparse
import contextlib
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
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ch10_vlm"))
OUT = ROOT / "outputs" / "ch16" / "p2"
GT_DIR = ROOT / "data" / "plate_gt"                                           # 번호판이 보이는 자료 — 저장소에 올리지 않는다
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--step", choices=["collect", "time", "candidates", "eval"], required=True)
ap.add_argument("--vlm-url", default="http://localhost:8000/v1")
ap.add_argument("--stride", type=int, default=3, help="몇 프레임마다 읽을까 (30fps 영상에서 3이면 초당 10번)")
args = ap.parse_args()
PW, PH = 520, 110                                                             # 정면화할 번호판 크기 (02-5, 04-6과 같음)
PLATE = re.compile(r"^[A-Z]{2}[0-9]{2} ?[A-Z]{3}$")                          # 영국 번호판 형식: 영문 2 + 숫자 2 + 영문 3


def load_function(path, names):
    """앞 장의 스크립트를 실행해 함수를 가져온다 (그 스크립트의 명령행 인자를 비워 두고)"""
    argv, sys.argv = sys.argv, [str(path)]
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    sys.argv = argv
    return [getattr(mod, n) for n in names]


def normalize(text):
    """OCR 결과를 형식에 맞춘다: 공백·기호를 지우고, 자리에 맞지 않는 글자를 헷갈리기 쉬운 짝으로 바꾼다 (O↔0, I↔1, S↔5, B↔8)"""
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(t) != 7:
        return t
    to_digit, to_alpha = str.maketrans("OIZSB", "01258"), str.maketrans("01258", "OIZSB")
    t = t[:2].translate(to_alpha) + t[2:4].translate(to_digit) + t[4:].translate(to_alpha)
    return f"{t[:4]} {t[4:]}"


# 1. 찾기·정면화·읽기 도구
def rule_plate(roi):
    """04-6의 Rule: 세로 경계가 촘촘하고 밝은 가로로 긴 사각형 → 네 꼭짓점 (차량 영역 좌표)"""
    found = find_plate(roi)
    return None if found is None else order_corners(found[1])


def gdino_plate(roi):
    """09-2의 Grounding DINO: 'license plate.'로 찾은 가장 점수가 높은 박스 → 네 꼭짓점"""
    image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    inp = gd_proc(images=image, text="license plate.", return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = gd(**inp)
    r = gd_proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[image.size[::-1]])[0]
    if len(r["boxes"]) == 0:
        return None
    x1, y1, x2, y2 = r["boxes"][r["scores"].argmax()].tolist()
    return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def frontal(img, quad):
    """02-5의 정면화: 네 꼭짓점을 PW x PH 직사각형으로 편다"""
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(np.float32(quad), np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]])), (PW, PH))


def read(plate_img):
    """번호판 자리는 이미 알고 있으므로 인식(rec)만 (16-4)"""
    r = ocr(plate_img, use_det=False, use_cls=False)
    return (r.txts[0], float(r.scores[0])) if r.txts else ("", 0.0)


if args.step in ("collect", "time"):
    find_plate, order_corners = load_function(ROOT / "ch04_detection" / "04_6_vehicle_plate.py", ["find_plate", "order_corners"])
    gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval().cuda()
    ocr = RapidOCR()
    yolo = YOLO("yolo26s.pt")
    cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
    records, n, t0 = [], 0, time.perf_counter()
    times, crops = defaultdict(float), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter()
        r = yolo.track(frame, persist=True, tracker="bytetrack.yaml", classes=[2, 5, 7], conf=0.4, verbose=False)[0]
        times["검출+추적"] += time.perf_counter() - t
        if n % args.stride == 0 and r.boxes.id is not None:
            for b, tid in zip(r.boxes.xyxy.int().tolist(), r.boxes.id.int().tolist()):
                x1, y1, x2, y2 = b
                if x2 - x1 < 150:                                             # 너무 작은 차량은 읽을 수 없다 (04-6)
                    continue
                roi = frame[y1:y2, x1:x2]
                if args.step == "time":
                    crops.append(roi.copy())
                    continue
                rec = {"frame": n, "track": tid, "width": x2 - x1}
                for name, fn in [("rule", rule_plate), ("gdino", gdino_plate)]:
                    t = time.perf_counter()
                    quad = fn(roi)
                    times[f"찾기 {name}"] += time.perf_counter() - t
                    if quad is None:
                        rec[name] = None
                        continue
                    t = time.perf_counter()
                    text, conf = read(frontal(roi, quad))
                    times["정면화+OCR"] += time.perf_counter() - t
                    rec[name] = {"quad": (np.float32(quad) + [x1, y1]).tolist(), "text": text, "conf": round(conf, 3)}
                records.append(rec)
        n += 1
    if args.step == "time":                                                   # 같은 함수를 차량 200개에 하나씩 돌려 평균
        sample = crops[::max(1, len(crops) // 200)][:200]
        for name, fn in [("Rule로 찾기", rule_plate), ("Grounding DINO로 찾기", gdino_plate)]:
            fn(sample[0])
            torch.cuda.synchronize()
            t = time.perf_counter()
            quads = [fn(c) for c in sample]
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) / len(sample) * 1000
            ok = [(c, q) for c, q in zip(sample, quads) if q is not None]
            t = time.perf_counter()
            for c, q in ok:
                read(frontal(c, q))
            print(f"{name}: {ms:.1f}ms/차량 | 정면화+OCR {(time.perf_counter() - t) / max(1, len(ok)) * 1000:.1f}ms/번호판 (찾은 {len(ok)}/{len(sample)})")
        print(f"검출+추적: 영상 {n}프레임 {times['검출+추적']:.0f}초 ({times['검출+추적'] / n * 1000:.1f}ms/프레임)")
        raise SystemExit
    (OUT / "frames.json").write_text(json.dumps(records), encoding="utf-8")
    print(f"영상 {n}프레임 ({time.perf_counter() - t0:.0f}초), 읽기 시도 {len(records)}번 (차량 {len({r['track'] for r in records})}대)")
    print("단계별 시간: " + " | ".join(f"{k} {v:.0f}초" for k, v in times.items()))
    raise SystemExit

records = json.loads((OUT / "frames.json").read_text(encoding="utf-8"))
by_track = defaultdict(list)
for r in records:
    by_track[r["track"]].append(r)


if args.step == "candidates":                                                 # 정답 만들기 도우미 — 사람이 사진을 보고 고르거나 적는다
    from vlm import ask
    GT_DIR.mkdir(parents=True, exist_ok=True)
    # 정답용 사진은 파이프라인의 결과와 무관하게: 차량 박스가 화면 안에 다 들어온 프레임 중 가장 큰 것에서 아래쪽 절반
    need = {(t, r["frame"]) for t, rs in by_track.items() for r in rs}
    yolo, seen = YOLO("yolo26s.pt"), defaultdict(list)
    cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = yolo.track(frame, persist=True, tracker="bytetrack.yaml", classes=[2, 5, 7], conf=0.4, verbose=False)[0]
        if r.boxes.id is not None:
            for b, tid in zip(r.boxes.xyxy.int().tolist(), r.boxes.id.int().tolist()):
                if (tid, n) in need and b[3] < frame.shape[0] - 5 and b[0] > 5 and b[2] < frame.shape[1] - 5:
                    seen[tid].append((b[2] - b[0], n, frame[(b[1] + b[3]) // 2:b[3], b[0]:b[2]].copy()))
        n += 1
    boxes = {t: max(v, key=lambda x: x[0]) for t, v in seen.items()}
    (GT_DIR / "gt_frames.json").write_text(json.dumps({t: v[1] for t, v in boxes.items()}), encoding="utf-8")
    cands, tiles = {}, []
    for tid, rs in sorted(by_track.items()):
        if tid not in boxes:
            continue
        crop = boxes[tid][2]
        votes = Counter(normalize(r[m]["text"]) for r in rs for m in ("rule", "gdino") if r[m] and r[m]["text"])
        vlm_read = ask("Read the vehicle registration plate in this image. Answer with the characters only.",
                       [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))], max_tokens=20, base_url=args.vlm_url)["text"]
        cands[tid] = list(dict.fromkeys([normalize(vlm_read)] + [t for t, _ in votes.most_common(3)]))
        k = min(360 / crop.shape[1], 180 / crop.shape[0])                     # 자르지 않고 칸에 맞춘다
        tile = np.full((180, 360, 3), 255, np.uint8)
        small = cv2.resize(crop, (int(crop.shape[1] * k), int(crop.shape[0] * k)))
        tile[:small.shape[0], :small.shape[1]] = small
        tile = cv2.copyMakeBorder(tile, 0, 30 * len(cands[tid]) + 30, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, f"track {tid}", (5, tile.shape[0] - 30 * len(cands[tid]) - 8), 0, 0.7, (0, 0, 255), 2)
        for i, c in enumerate(cands[tid]):
            cv2.putText(tile, f"{i}: {c}", (5, tile.shape[0] - 30 * (len(cands[tid]) - i) + 20), 0, 0.7, (0, 0, 0), 2)
        tiles.append(cv2.resize(tile, (360, 300)))
    (GT_DIR / "candidates.json").write_text(json.dumps(cands), encoding="utf-8")
    for i in range(0, len(tiles), 12):
        page = tiles[i:i + 12] + [np.full_like(tiles[0], 255)] * (12 - len(tiles[i:i + 12]))
        cv2.imwrite(str(GT_DIR / f"candidates_{i // 12}.jpg"), np.vstack([np.hstack(page[j:j + 4]) for j in range(0, 12, 4)]))
    print(f"후보 {len(cands)}대 (추적 {len(by_track)}대 중 박스를 다시 찾은 차량) → {GT_DIR}")
    print("사진을 보고 gt.json에 {track: 맞는 후보 번호, 맞는 후보가 없으면 사람이 읽은 글자, 읽을 수 없으면 null}을 적는다")
    raise SystemExit

# 3. 평가 — 정답과 비교 (번호판 글자는 출력하지 않는다)
from vlm import ask  # noqa: E402

picks = json.loads((GT_DIR / "gt.json").read_text(encoding="utf-8"))
cands = json.loads((GT_DIR / "candidates.json").read_text(encoding="utf-8"))
def answer(v):
    """후보 번호(int), 다른 추적 ID와 같은 번호판("@95"), 또는 사람이 읽은 글자"""
    if isinstance(v, int):
        return v
    return answer(picks[v[1:]]) if v.startswith("@") else v


gt = {}
for t, v in picks.items():
    if v is None:
        continue
    a = answer(v)
    src = v[1:] if isinstance(v, str) and v.startswith("@") else t
    gt[int(t)] = cands[src][a] if isinstance(a, int) else normalize(a)
gt_frame = {int(t): f for t, f in json.loads((GT_DIR / "gt_frames.json").read_text(encoding="utf-8")).items()}
at_gt = {t: next(r for r in by_track[t] if r["frame"] == gt_frame[t]) for t in gt}      # 정답을 만든 그 프레임의 파이프라인 결과
print(f"정답이 있는 차량 {len(gt)}대 (번호판이 보이고 사람이 읽을 수 있던 차량), 읽기 시도 {sum(len(by_track[t]) for t in gt)}번")


def cer(a, b):
    """글자 오류율: 편집 거리 ÷ 정답 길이 (공백 제외)"""
    a, b = a.replace(" ", ""), b.replace(" ", "")
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1] / max(1, len(b))


print(f"\n{'방법':<16}{'번호판 찾음':>10}{'한 프레임 정확':>14}{'+형식 규칙':>11}{'CER':>7}{'다수결(차량)':>13}")
for m in ["rule", "gdino"]:
    tries = [(r, gt[t]) for t in gt for r in by_track[t]]
    found = [(r[m], g) for r, g in tries if r[m]]
    raw = np.mean([r["text"].replace(" ", "").upper() == g.replace(" ", "") for r, g in found])
    fixed = [(normalize(r["text"]), g) for r, g in found]
    vote = []
    for t in gt:
        texts = [normalize(r[m]["text"]) for r in by_track[t] if r[m]]
        valid = [x for x in texts if PLATE.match(x)]
        vote.append(Counter(valid or texts).most_common(1)[0][0] == gt[t] if texts else False)
    print(f"{m:<16}{len(found) / len(tries):>10.1%}{raw:>14.1%}{np.mean([a == g for a, g in fixed]):>11.1%}"
          f"{np.mean([cer(a, g) for a, g in fixed]):>7.2f}{np.mean(vote):>13.1%}")

# 오류 유형 — 정답을 만든 프레임(차량이 화면 안에서 가장 크게 보인 순간) 하나에서, 방법마다
print()
for m in ["rule", "gdino"]:
    kinds = Counter()
    for t, g in gt.items():
        hit = at_gt[t][m]
        if hit is None:
            kinds["번호판 못 찾음"] += 1
        elif len(re.sub(r"[^A-Z0-9]", "", hit["text"].upper())) < 5:
            kinds["찾았지만 글자를 거의 못 읽음 (위치·정면화 문제)"] += 1
        elif normalize(hit["text"]) != g:
            kinds["읽었지만 틀림 (OCR 오인식)"] += 1
        else:
            kinds["맞음"] += 1
    print(f"오류 유형 [{m}]: " + " | ".join(f"{k} {v}" for k, v in kinds.most_common()))

# VLM 검증 — 다수결이 형식에 안 맞거나 표가 갈린 차량만 VLM(Qwen3.5)이 다시 읽는다
cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
final, vote_only, asked, t0 = {}, {}, [], time.perf_counter()
for t in gt:
    texts = [normalize(r[m]["text"]) for r in by_track[t] for m in ("rule", "gdino") if r[m]]
    votes = Counter(x for x in texts if PLATE.match(x))
    top = votes.most_common(2)
    vote_only[t] = top[0][0] if top else ""                                   # 비교용: VLM 없이 두 방법을 합친 다수결만
    sure = top and top[0][1] >= 3 and (len(top) == 1 or top[0][1] >= 2 * top[1][1])
    if sure:
        final[t] = top[0][0]
        continue
    b = at_gt[t]
    hit = b["gdino"] or b["rule"]                                             # 더 잘 찾는 Grounding DINO의 위치를 먼저
    if hit is None:
        final[t] = ""
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, b["frame"])
    frame = cap.read()[1]
    q = np.float32(hit["quad"])
    x1, y1 = np.maximum(q.min(0).astype(int) - 40, 0)
    x2, y2 = q.max(0).astype(int) + 40
    v = ask("Read the vehicle registration plate in this image. Answer with the characters only.",
            [Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))], max_tokens=20, base_url=args.vlm_url)["text"]
    final[t] = normalize(v)
    asked.append(t)
print(f"\n추적 ID {len(gt)}개 (서로 다른 번호판 {len(set(gt.values()))}개 — 같은 차가 ID 둘 이상으로 잡힌 경우가 있다)")
print(f"  두 방법을 합친 다수결만: 정확 {np.mean([vote_only[t] == gt[t] for t in gt]):.1%}")
print(f"  다수결이 확실하면 그대로, 애매하면 VLM: 정확 {np.mean([final[t] == gt[t] for t in gt]):.1%} "
      f"| VLM 호출 {len(asked)}번 ({time.perf_counter() - t0:.1f}초)")
print(f"  VLM을 부른 {len(asked)}대: 다수결만으로 맞은 것 {sum(vote_only[t] == gt[t] for t in asked)}대 → VLM 답이 맞은 것 {sum(final[t] == gt[t] for t in asked)}대")
