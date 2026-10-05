"""12-2. VLM 평가하기 — 정답이 하나가 아닌 답을 어떻게 채점할까

1) 정답 매칭: "몇 명인가?"를 형식 지시 없이 묻고 세 가지로 채점 — 문자열 일치 / 규칙(숫자 꺼내기) / LLM 채점
2) Vision Hallucination: 설명 문장에서 COCO 물체 이름을 찾아 정답과 대조 (CHAIR) — 한 문장 vs 자세히
3) LLM 채점: 사람이 쓴 설명 5개를 기준으로 VLM의 설명에 1~5점 — CHAIR가 찾은 "지어낸 물체"를 잡아내는가

실행 (저장소 루트에서, 서버를 띄운 뒤. 채점 모델은 --judge-url, 생략하면 같은 서버):
    python ch12_evaluation/12_2_vlm_eval.py
    python ch12_evaluation/12_2_vlm_eval.py --base-url http://localhost:8001/v1 --judge-url http://localhost:8000/v1
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ch10_vlm"))
from vlm import ask, model_name  # noqa: E402

COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
OUT = ROOT / "outputs" / "ch12"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--judge-url", default="", help="채점에 쓸 서버 (생략하면 --base-url)")
ap.add_argument("--n-count", type=int, default=100)
ap.add_argument("--n-caption", type=int, default=150)
args = ap.parse_args()
B, J = args.base_url, args.judge_url or args.base_url
M = model_name(B).split("/")[-1]
print(f"평가 대상: {M} | 채점 모델: {model_name(J).split('/')[-1]}")
inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
caps = json.loads((COCO_DIR / "captions.json").read_text(encoding="utf-8"))["annotations"]
names = {c["id"]: c["name"] for c in inst["categories"]}
anns, refs = defaultdict(list), defaultdict(list)
for a in inst["annotations"]:
    anns[a["image_id"]].append(a)
for c in caps:
    refs[c["image_id"]].append(c["caption"].strip())
load = lambda im: Image.open(COCO_DIR / "images" / im["file_name"])  # noqa: E731
log = {}

# 1. 정답 매칭 — 형식을 정해 주지 않은 답
WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                     "fifteen sixteen seventeen eighteen nineteen twenty".split())}


def rule_number(text):
    """답에서 처음 나오는 숫자(아라비아 숫자 또는 영어 수 단어)"""
    m = re.search(r"\b(\d+|" + "|".join(WORDS) + r")\b", text.lower())
    return None if m is None else int(m.group(1)) if m.group(1).isdigit() else WORDS[m.group(1)]


def judge_same(gt, answer):
    q = (f"Question: How many people are in this image?\nReference answer: {gt}\nModel answer: {answer}\n"
         "Does the model answer state the same number of people as the reference? Answer yes or no.")
    return ask(q, max_tokens=4, base_url=J)["text"].strip().lower().startswith("yes")


people = [im for im in inst["images"] if any(a["category_id"] == 1 for a in anns[im["id"]])
          and not any(a["category_id"] == 1 and a["iscrowd"] for a in anns[im["id"]])][:args.n_count]
rows = []
for im in people:
    gt = sum(a["category_id"] == 1 for a in anns[im["id"]])
    ans = ask("How many people are in this image?", [load(im)], max_tokens=64, base_url=B)["text"].strip()
    rows.append((im["id"], gt, ans, ans.rstrip(".").strip() == str(gt), rule_number(ans) == gt, judge_same(gt, ans)))
log["count"] = rows
print(f"\n[정답 매칭] 사람 수 {len(rows)}장, 형식 지시 없이 \"How many people are in this image?\"")
for i, label in [(3, "문자열 일치"), (4, "규칙: 처음 나오는 숫자"), (5, "LLM 채점")]:
    print(f"  {label:<14} 정확도 {np.mean([r[i] for r in rows]):.1%}")
print(f"  규칙과 LLM 채점이 같은 판정: {np.mean([r[4] == r[5] for r in rows]):.1%}")
lengths = [len(r[2]) for r in rows]
print(f"  답의 길이: 중앙값 {np.median(lengths):.0f}자, 숫자만 답한 비율 {np.mean([r[2].rstrip('.').isdigit() for r in rows]):.1%}")
for r in [r for r in rows if r[4] != r[5]][:4]:
    print(f"  규칙 {'O' if r[4] else 'X'} / LLM {'O' if r[5] else 'X'} | 정답 {r[1]} | 답: {r[2][:110]}")

# 2. CHAIR — 설명에 나온 COCO 물체가 실제로 있는가
SYN = {"person": "person people man men woman women boy boys girl girls child children kid kids guy guys lady ladies player players "
                 "skier skiers surfer surfers skateboarder rider riders crowd",
       "bicycle": "bicycle bicycles bike bikes", "car": "car cars", "motorcycle": "motorcycle motorcycles motorbike",
       "airplane": "airplane airplanes plane planes jet jets aircraft", "tv": "tv tvs television", "couch": "couch couches sofa",
       "cell phone": "cell phone|cellphone|phone|phones|smartphone", "dining table": "dining table|table|tables",
       "potted plant": "potted plant|houseplant", "sports ball": "ball|balls", "cup": "cup cups mug mugs",
       "donut": "donut donuts doughnut doughnuts", "laptop": "laptop laptops", "remote": "remote|remote control|remotes",
       "handbag": "handbag handbags purse", "baseball glove": "baseball glove|glove|mitt", "bus": "bus buses",
       "bench": "bench benches", "knife": "knife knives", "sandwich": "sandwich sandwiches", "wine glass": "wine glass|wine glasses",
       "orange": "oranges"}                                                   # 색 이름 "orange"와 헷갈리지 않게 복수형만
PHRASES = []                                                                  # (찾을 말, 클래스) — 긴 말부터 찾는다
for cls in names.values():
    words = SYN.get(cls)
    variants = (words.split("|") if "|" in words else words.split()) if words else [cls, cls + "s"]
    PHRASES += [(w, cls) for w in variants]
PHRASES.sort(key=lambda x: -len(x[0]))


NEG = {"no", "not", "without", "nor", "neither", "never", "absent"}
OTHER_SENSE = {"apple": {"computer", "logo", "imac", "macbook"}, "car": {"of"}, "mouse": {"pad"}}   # "apple computer", "car of the train"


def mentioned(text):
    """설명에 나온 COCO 물체. 부정("no boats")과 다른 뜻("apple computer")은 뺀다"""
    found = set()
    for clause in re.split(r"[\n.;:!?]", text.lower()):                     # 문장(절) 단위로
        words = re.sub(r"[^a-z ]", " ", clause).split()
        for w, cls in PHRASES:                                                # 긴 말부터 — "hot dog"을 "dog"으로 다시 세지 않도록
            n = len(w.split())
            for k in range(len(words) - n + 1):
                if words[k:k + n] != w.split():
                    continue
                before, after = words[max(0, k - 4):k], words[k + n:k + n + 4]
                if not (NEG & set(before) or NEG & set(after) or set(after[:1]) & OTHER_SENSE.get(w, set())):
                    found.add(cls)
                words[k:k + n] = ["|"] * n
    return found


targets = inst["images"][:args.n_caption]
chair = {}
for key, prompt, mt in [("short", "Describe this image in one sentence.", 80), ("detail", "Describe this image in detail.", 400)]:
    hall_obj = all_obj = hall_cap = covered = gt_total = 0
    out = []
    for im in targets:
        text = ask(prompt, [load(im)], max_tokens=mt, base_url=B)["text"].strip()
        have = {names[a["category_id"]] for a in anns[im["id"]]}
        found = mentioned(text)
        fake = found - have
        hall_obj, all_obj, hall_cap = hall_obj + len(fake), all_obj + len(found), hall_cap + bool(fake)
        covered, gt_total = covered + len(found & have), gt_total + len(have)
        out.append({"image_id": im["id"], "text": text, "fake": sorted(fake), "found": sorted(found)})
    chair[key] = out
    print(f"\n[CHAIR] {prompt} ({len(targets)}장, 평균 {np.mean([len(o['text']) for o in out]):.0f}자)")
    print(f"  CHAIR_i (언급한 물체 중 없는 것) {hall_obj / max(1, all_obj):.1%} | CHAIR_s (없는 물체가 하나라도 있는 설명) "
          f"{hall_cap / len(targets):.1%} | 정답 물체를 언급한 비율 {covered / gt_total:.1%}")
log["chair"] = chair
fake_count = defaultdict(int)
for o in chair["detail"]:
    for f in o["fake"]:
        fake_count[f] += 1
print("  자세히 설명할 때 가장 자주 지어낸 물체: " + ", ".join(f"{k} {v}회" for k, v in sorted(fake_count.items(), key=lambda x: -x[1])[:6]))


# 3. LLM 채점 — 사람의 설명 5개를 기준으로
def judge_score(ref, text):
    q = ("Here are five descriptions of an image written by different people:\n" + "\n".join(f"- {r}" for r in ref) +
         f"\n\nCandidate description:\n{text}\n\nRate the factual accuracy of the candidate from 1 to 5. "
         "5 = everything in it is consistent with the image as described by the people, "
         "1 = it mentions several things that are not in the image. Answer with the number only.")
    return rule_number(ask(q, max_tokens=4, base_url=J)["text"])


scores = []
for o in chair["detail"]:
    s = judge_score(refs[o["image_id"]], o["text"])
    o["judge"] = s
    scores.append((s, bool(o["fake"]), len(o["text"])))
ok = [s for s, f, _ in scores if s is not None]
print(f"\n[LLM 채점] 자세한 설명 {len(scores)}개, 점수 평균 {np.mean(ok):.2f}")
for flag, label in [(False, "CHAIR가 깨끗하다고 본 설명"), (True, "CHAIR가 없는 물체를 찾은 설명")]:
    sel = [s for s, f, _ in scores if f == flag and s is not None]
    print(f"  {label:<20} {len(sel):>3}개: 평균 {np.mean(sel):.2f}점, 5점 {np.mean([s == 5 for s in sel]):.0%}, 4점 이상 {np.mean([s >= 4 for s in sel]):.0%}")
print(f"  설명 길이와 점수의 상관계수 {np.corrcoef([l for s, _, l in scores if s], [s for s, _, _ in scores if s])[0, 1]:.2f}")
for o in [o for o in chair["detail"] if o["fake"] and (o["judge"] or 0) >= 4][:4]:
    print(f"  [{o['image_id']}] 지어냈다고 본 물체 {o['fake']} → LLM {o['judge']}점")
(OUT / f"12_2_{M}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
