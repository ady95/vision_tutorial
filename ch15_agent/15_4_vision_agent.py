"""15-4. 실습: 보고 판단하고 행동하는 Agent — Camera → Detection → VLM → Decision → Action

05-5의 교통 영상(60초)을 1초에 한 장씩 보며 "갓길(오른쪽 실선 밖)에 차량이 있으면 운영자에게 알린다".
정답: 경찰차가 경광등을 켜고 0~23초 동안 오른쪽 갓길을 달린다 (저자가 프레임을 보고 확인). 24~59초는 갓길이 비어 있다.

행동은 도구 3개로만 할 수 있다:
    log_event(기록만) / alert_operator(운영자 알림) / request_lane_closure(차선 차단 요청 — 사람 확인 전에는 실행하지 않음)
안전 장치: 행동 목록 제한, 영향이 큰 행동은 사람 확인, 같은 사건의 알림은 10초에 한 번

구성 3가지를 비교한다:
    A. VLM만: 매 초 전체 화면을 VLM이 보고 행동을 고른다
    B. YOLO → Rule(갓길 영역) → VLM: 갓길 후보가 있는 초만 VLM이 확인하고 행동을 고른다
    C. YOLO → Rule: 후보가 있으면 바로 알린다

실행 (저장소 루트에서, 15-2와 같이 도구 호출을 켠 서버를 띄운 뒤):
    python ch15_agent/15_4_vision_agent.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ch10_vlm"))
from vlm import client, encode, model_name  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
args = ap.parse_args()
B = args.base_url
GT = {s: s <= 23 for s in range(60)}                                         # 정답: 0~23초에 갓길에 경찰차
SHOULDER = np.array([(1240, 300), (1300, 300), (1600, 600), (1920, 760), (1920, 1080),     # 오른쪽 갓길 (1920x1080 화소 좌표)
                     (1680, 1080), (1500, 800), (1360, 560), (1280, 370)], np.int32)
VEHICLES = {"car", "truck", "bus", "motorcycle"}

# 1. 카메라 — 1초에 한 장
cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
for s in range(60):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
    frames.append(cap.read()[1])

# 2. 인식(YOLO) + Rule — 바닥 중심이 갓길 영역 안에 있는 차량
yolo = YOLO("yolo26s.pt")


def shoulder_candidates(frame):
    r = yolo(frame, conf=0.25, verbose=False)[0]
    out = []
    for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
        foot = ((b[0] + b[2]) / 2, b[3])                                      # 차가 땅에 닿은 점
        if r.names[int(c)] in VEHICLES and cv2.pointPolygonTest(SHOULDER, foot, False) >= 0:
            out.append(b)
    return out


# 3. 판단(VLM)과 행동 — 도구로만 행동한다
ACTIONS = [
    ("log_event", "Record an observation only. Use when no vehicle is on the shoulder or nothing needs attention.",
     {"note": {"type": "string"}}),
    ("alert_operator", "Notify the human operator that a vehicle is on the road shoulder.",
     {"message": {"type": "string"}, "level": {"type": "string", "enum": ["info", "warning"]}}),
    ("request_lane_closure", "Ask to close a traffic lane. Only for an accident or a vehicle blocking a lane.",
     {"reason": {"type": "string"}}),
]
TOOLS = [{"type": "function", "function": {"name": n, "description": d,
                                          "parameters": {"type": "object", "properties": p, "required": list(p)}}} for n, d, p in ACTIONS]
PROMPT = ("You are the monitoring agent of a highway camera. The road shoulder is the strip to the right of the solid white line "
          "at the right edge of the road. {context} Decide what to do by calling exactly one tool: "
          "alert_operator if a vehicle is actually on the shoulder (level 'info' for an emergency vehicle with flashing lights, "
          "'warning' for a stopped or broken-down vehicle), log_event if not.")


def human_confirm(action, arg):
    """영향이 큰 행동은 사람이 확인해야 실행된다. 이 실습에서는 사람이 없으므로 늘 보류"""
    return False


def decide(images, context):
    content = [{"type": "image_url", "image_url": {"url": encode(im, max_side=1280)}} for im in images]
    content.append({"type": "text", "text": PROMPT.format(context=context)})
    r = client(B).chat.completions.create(model=model_name(B), messages=[{"role": "user", "content": content}], max_tokens=200,
                                          temperature=0.0, tools=TOOLS, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    calls = r.choices[0].message.tool_calls or []
    if not calls:
        return "log_event", {"note": "(도구를 부르지 않음)"}
    return calls[0].function.name, json.loads(calls[0].function.arguments or "{}")


def run(mode):
    """초마다 (판단, 실제로 실행된 행동) 기록"""
    log, last_alert, vlm_calls, blocked = [], -99, 0, 0
    t = time.perf_counter()
    for s, frame in enumerate(frames):
        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if mode == "A":
            action, arg = decide([rgb], "Look at the camera image.")
            vlm_calls += 1
        else:
            cands = shoulder_candidates(frame)
            if not cands:
                action, arg = "log_event", {"note": "no candidate"}
            elif mode == "C":
                action, arg = "alert_operator", {"message": "vehicle in shoulder area", "level": "info"}
            else:
                x1, y1, x2, y2 = cands[0]
                w, h = x2 - x1, y2 - y1
                crop = rgb.crop((max(0, x1 - w), max(0, y1 - h), min(1920, x2 + w), min(1080, y2 + h)))
                action, arg = decide([rgb, crop], "A detector flagged a vehicle whose position is in the shoulder area. "
                                                  "The first image is the full frame, the second is a close-up of the flagged vehicle.")
                vlm_calls += 1
        executed = action
        if action == "request_lane_closure" and not human_confirm(action, arg):
            executed, blocked = "lane_closure_held", blocked + 1                 # 사람 확인 전에는 실행하지 않음
        if executed == "alert_operator":
            if s - last_alert < 10:
                executed = "alert_suppressed"                                 # 같은 사건의 알림은 10초에 한 번
            else:
                last_alert = s
        log.append((s, action, executed, arg))
    sec = time.perf_counter() - t
    said = {s: a in ("alert_operator", "request_lane_closure") for s, a, _, _ in log}
    tp = sum(said[s] and GT[s] for s in GT)
    fp = sum(said[s] and not GT[s] for s in GT)
    fn = sum(not said[s] and GT[s] for s in GT)
    alerts = [s for s, _, e, _ in log if e == "alert_operator"]
    print(f"{mode}: 판단 정확도 {(60 - fp - fn) / 60:.1%} (놓침 {fn}초, 헛경보 {fp}초) | 실제 알림 {len(alerts)}번 {alerts} | "
          f"VLM {vlm_calls}번 | 차선 차단 요청 {blocked}번(보류) | {sec:.1f}초")
    return log


print(f"정답: 0~23초 갓길에 경찰차 (24초), 24~59초 없음 (36초)")
logs = {m: run(m) for m in ["C", "B", "A"]}
cand = [s for s in range(60) if shoulder_candidates(frames[s])]
print(f"\nYOLO+Rule 후보가 있던 초: {cand}")
for m in ["A", "B"]:
    wrong = [(s, a, arg) for s, a, _, arg in logs[m] if (a != "log_event") != GT[s]]
    print(f"\n[{m}] 틀린 판단 {len(wrong)}개")
    for s, a, arg in wrong:
        print(f"  {s}초 정답 {'갓길에 차' if GT[s] else '없음'} → {a} {json.dumps(arg, ensure_ascii=False)[:150]}")
