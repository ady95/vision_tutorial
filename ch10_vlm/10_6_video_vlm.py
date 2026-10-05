"""10-6. 실습: Multi-image와 Video VLM

1) 여러 이미지 비교 — 05-5 영상의 6초 간격 두 프레임, 같은 차인지 다른 차인지
2) 영상 이해 — 05-5의 공개 교통 영상을 통째로 넣고 묻는다 (정답이 있는 질문: 다가오는 차 49대, 경찰차가 화면을 벗어나는 시점)

영상은 vLLM에 data URL(mp4)로 보낸다. 토큰을 줄이려고 가로 640, 초당 2프레임으로 줄인 사본을 만들어 보낸다.

실행 (저장소 루트에서, vLLM 서버를 띄운 뒤):
    python ch10_vlm/10_6_video_vlm.py
    python ch10_vlm/10_6_video_vlm.py --same 내_차_옆.jpg 내_차_뒤.jpg --other 다른_차.jpg
"""
import argparse
import base64
import sys
import tempfile
import time
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm import ask, client, model_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "videos" / "traffic.mp4"
ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--same", nargs=2, default=[], help="같은 차를 다른 방향에서 찍은 사진 두 장")
ap.add_argument("--other", default="", help="다른 차 사진 한 장")
args = ap.parse_args()
B = args.base_url
print(f"모델: {model_name(B)}")


def frame_at(n):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    return Image.fromarray(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))


# 1. 여러 이미지 비교
a, b = frame_at(1380), frame_at(1560)
for prompt in ["These two frames are from a fixed traffic camera, taken 6 seconds apart. What changed between them? Answer in two sentences.",
               "In which of the two images is the white van with a roof rack closer to the camera, the first or the second? Answer 'first' or 'second' and explain briefly."]:
    r = ask(prompt, [a, b], max_tokens=200, base_url=B, max_side=1280)          # 1920x1080 두 장은 12GB GPU에서 메모리 부족
    print(f"[프레임 1380, 1560] {prompt[:60]}...\n  → {r['text']}  ({r['seconds']:.2f}초, 입력 {r['prompt_tokens']} 토큰)")
if args.same and args.other:
    imgs = [Image.open(p) for p in args.same]
    for pair, truth in [((imgs[0], imgs[1]), "같은 차"), ((imgs[0], Image.open(args.other)), "다른 차")]:
        r = ask("Do these two photos show the same car (the main car in each photo)? Answer 'same' or 'different' first, "
                "then give one reason. Do not mention license plate characters.", list(pair), max_tokens=120, base_url=B, max_side=1280)
        print(f"[정답: {truth}] → {r['text']}  ({r['seconds']:.2f}초)")


# 2. 영상 — 가로 640, 초당 2프레임 사본을 만들어 통째로 보낸다
def small_clip(src, fps=2, width=640):
    cap = cv2.VideoCapture(str(src))
    step = round(cap.get(cv2.CAP_PROP_FPS) / fps)
    out_path = Path(tempfile.gettempdir()) / f"clip_{fps}fps_{width}.mp4"
    writer, n = None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if n % step == 0:
            f = cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
            writer.write(f)
        n += 1
    writer.release()
    return out_path


clip = small_clip(VIDEO)
url = "data:video/mp4;base64," + base64.b64encode(clip.read_bytes()).decode()
print(f"\n영상 사본: {clip.stat().st_size / 1e6:.1f} MB (가로 640, 초당 2프레임, 60초)")
QUESTIONS = [("Describe what happens in this traffic video in two sentences.", None),
             ("How many vehicles drive toward the camera during this video? Answer with a number only.", "정답 49 (05-5에서 슬릿 스캔으로 센 값)"),
             ("A police car with flashing blue lights drives along the right shoulder. At what second does it leave the bottom of the frame? "
              "Answer with a number of seconds only.", "정답: 21~24초 (21초에 화면 오른쪽 아래, 24초에는 없음)"),
             ("Is the traffic in the lanes coming toward the camera moving freely or congested? Answer in one sentence.", "정답: 정체")]
for q, truth in QUESTIONS:
    t = time.perf_counter()
    r = client(B).chat.completions.create(
        model=model_name(B), max_tokens=200, temperature=0.0,
        messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": url}}, {"type": "text", "text": q}]}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    print(f"[영상] {q}\n  → {r.choices[0].message.content.strip()}  ({time.perf_counter() - t:.1f}초, 입력 {r.usage.prompt_tokens} 토큰)"
          + (f"\n  ({truth})" if truth else ""))
