"""15-2. VLM + Tool Calling — Qwen3.5-9B가 Vision 도구를 골라 부르게 하기

도구 3개: detect_objects(YOLO26s), measure_part(02-6의 Rule 측정), read_text(RapidOCR)
1) bus.jpg로 대화 하나를 따라가 본다 — 어떤 도구를 어떤 인자로 부르고, 결과로 어떻게 답하나
2) 부품 평가셋(12-3, 420장): 도구 없이 vs 도구와 함께 — 12-3에서 9B가 못 잡은 위치·크기 불량
3) COCO 사람 세기(100장): 검출 도구가 있으면 VLM은 도구의 숫자를 얼마나 따르나

서버는 도구 호출을 켜서 띄운다 (Qwen3.5의 도구 호출 형식은 qwen3_coder 파서):
    vllm serve Qwen/Qwen3.5-9B --port 8000 --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.92 \\
        --enable-auto-tool-choice --tool-call-parser qwen3_coder
실행 (저장소 루트에서, 12-3 실행 후):
    python ch15_agent/15_2_vlm_tool_calling.py
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
from PIL import Image
from rapidocr import RapidOCR
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ch10_vlm"))
from vlm import client, encode, first_int, model_name  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
args = ap.parse_args()
B = args.base_url

# 1. 도구 — 이미지는 Agent가 지금 보고 있는 것 하나. 도구는 그 이미지에 대해 숫자와 글을 돌려준다
spec = importlib.util.spec_from_file_location("rule", ROOT / "ch02_rule_based" / "02_6_rule_inspection.py")
rule = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(rule)                                             # 02-6의 측정 함수와 기준값(spec)
yolo, ocr = YOLO("yolo26s.pt"), RapidOCR()


def detect_objects(path):
    r = yolo(path, conf=0.25, verbose=False)[0]
    counts = Counter(r.names[int(c)] for c in r.boxes.cls)
    return {"counts": dict(counts), "note": "YOLO26s, confidence >= 0.25, 80 COCO classes only"}


def measure_part(path):
    f = rule.measure(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), True)
    if f is None:
        return {"error": "no part found"}
    s = rule.spec
    return {"area_ratio_to_normal": round(f["area"] / s["area"], 3), "holes": f["holes"], "corners": f["corners"],
            "angle_deg": round(f["angle"], 1), "offset_from_center_px": round(f["offset"], 1),
            "solidity": round(f["solidity"], 3), "compactness": round(f["compactness"], 2),
            "limits": {"area_ratio": "0.94~1.06", "holes": 2, "corners": 4, "angle_deg": "-8~8", "offset_from_center_px": "<= 40",
                       "solidity": f">= {s['solidity_min']:.3f}", "compactness": f"<= {s['compactness_max']:.2f}"}}


def read_text(path):
    r = ocr(str(path))
    return {"texts": [t for t, sc in zip(r.txts or [], r.scores or []) if sc >= 0.8]}


TOOLS = {"detect_objects": detect_objects, "measure_part": measure_part, "read_text": read_text}
SCHEMAS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": {}}}}
           for n, d in [("detect_objects", "Detect objects in the current image with an object detector (80 COCO classes). Returns counts per class."),
                        ("measure_part", "Measure the metal bracket in the current image (size, holes, corners, angle, offset) and return the inspection limits."),
                        ("read_text", "Read the text in the current image with OCR.")]]


def agent(question, path, tools=True, max_rounds=4):
    """도구를 부르며 답할 때까지 반복. (답, 부른 도구 목록)"""
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": encode(Image.open(path))}},
                                         {"type": "text", "text": question}]}]
    called = []
    for _ in range(max_rounds):
        r = client(B).chat.completions.create(model=model_name(B), messages=msgs, max_tokens=512, temperature=0.0,
                                              tools=SCHEMAS if tools else None,
                                              extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        m = r.choices[0].message
        if not m.tool_calls:
            return (m.content or "").strip(), called
        msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
        for tc in m.tool_calls:
            called.append(tc.function.name)
            result = TOOLS[tc.function.name](path) if tc.function.name in TOOLS else {"error": "unknown tool"}
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})
    return "(도구 호출 횟수 초과)", called


# 2. 대화 하나 따라가기
bus = ROOT / "data" / "images" / "bus.jpg"
for q in ["How many people are in this image, and what is written on the bus? Use the tools if they help.",
          "Is this a metal bracket with a defect?"]:
    t = time.perf_counter()
    ans, called = agent(q, bus)
    print(f"[bus.jpg] {q}\n  부른 도구: {called or '없음'} ({time.perf_counter() - t:.1f}초)\n  → {ans[:300]}")

# 3. 부품 평가셋 — 도구 없이 vs 도구와 함께
Q = ("This is a photo of a metal bracket on a conveyor belt. A good bracket is a rounded rectangle with exactly two round holes, "
     "no broken corners, no cracks, placed straight in the center, and of the normal size. Is this bracket defective? "
     "Use the tools if they help. Finally answer with 'ok' or 'defect' only.")
with open(ROOT / "data" / "parts_bench" / "labels.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"\n부품 평가셋 {len(rows)}장")
for use_tools in [False, True]:
    per, calls, t = defaultdict(list), Counter(), time.perf_counter()
    for r in rows:
        ans, called = agent(Q, ROOT / "data" / "parts_bench" / r["file"], tools=use_tools)
        pred = bool(re.search(r"\bdefect", ans.lower()))
        per[r["label"]].append(pred == (r["label"] != "ok"))
        calls.update(called)
        calls["(도구를 하나라도 부른 사진)"] += bool(called)
    acc = np.mean([v for vs in per.values() for v in vs])
    print(f"  {'도구와 함께' if use_tools else '도구 없이':<8} 정확도 {acc:.1%} | " + " ".join(f"{k} {np.mean(v):.0%}" for k, v in per.items())
          + f" | {(time.perf_counter() - t) / len(rows):.2f}초/장" + (f" | 호출 {dict(calls)}" if use_tools else ""))

# 4. COCO 사람 세기 — 도구의 숫자를 따르나
inst = json.loads((ROOT / "data" / "datasets" / "coco_val500" / "instances.json").read_text(encoding="utf-8"))
anns = defaultdict(list)
for a in inst["annotations"]:
    anns[a["image_id"]].append(a)
imgs = [im for im in inst["images"] if any(a["category_id"] == 1 for a in anns[im["id"]])
        and not any(a["category_id"] == 1 and a["iscrowd"] for a in anns[im["id"]])][:100]
q = "How many people are in this image? Use the tools if they help. Finally answer with a number only."
res = defaultdict(list)
for im in imgs:
    path = ROOT / "data" / "datasets" / "coco_val500" / "images" / im["file_name"]
    gt = sum(a["category_id"] == 1 for a in anns[im["id"]])
    tool_n = detect_objects(path)["counts"].get("person", 0)
    for use_tools in [False, True]:
        ans, called = agent(q, path, tools=use_tools)
        n = first_int(ans.split("\n")[-1]) if ans else None
        res[use_tools].append((gt, n, tool_n, bool(called)))
print(f"\nCOCO 사람 세기 {len(imgs)}장 (YOLO26s 개수의 정확도 {np.mean([g == t for g, _, t, _ in res[True]]):.1%})")
for use_tools, rs in res.items():
    print(f"  {'도구와 함께' if use_tools else '도구 없이':<8} 정확도 {np.mean([g == n for g, n, _, _ in rs]):.1%} | "
          f"YOLO 개수와 같은 답 {np.mean([n == t for _, n, t, _ in rs]):.1%}" +
          (f" | 도구를 부른 비율 {np.mean([c for *_, c in rs]):.1%}" if use_tools else ""))
