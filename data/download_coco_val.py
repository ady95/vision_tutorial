"""COCO val2017에서 앞 N장(기본 500장)과 정답만 내려받기 (06-2 Detection 비교)

1) 정답 파일 annotations_trainval2017.zip(약 240MB)에서 instances_val2017.json만 꺼내
2) 이미지 번호가 작은 순서로 N장을 골라 그 이미지만 내려받고
3) 정답도 그 N장 것만 남겨 data/datasets/coco_val500/instances.json 으로 저장합니다.
정답(annotations)은 CC BY 4.0, 이미지는 Flickr 원저작자의 라이선스를 따릅니다.

실행:
    python data/download_coco_val.py            # 500장
    python data/download_coco_val.py --n 1000
"""
import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
args = ap.parse_args()
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"
DEST = Path(__file__).resolve().parent / "datasets"
OUT = DEST / f"coco_val{args.n}"
(OUT / "images").mkdir(parents=True, exist_ok=True)


def fetch(url, path):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vision-tutorial"}), timeout=60) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)


full = DEST / "instances_val2017.json"
if not full.exists():
    zpath = DEST / "annotations_trainval2017.zip"
    print(f"정답 파일 내려받는 중: {ANN_URL}")
    fetch(ANN_URL, zpath)
    with zipfile.ZipFile(zpath) as z, z.open("annotations/instances_val2017.json") as src, open(full, "wb") as dst:
        shutil.copyfileobj(src, dst)
    zpath.unlink()

coco = json.loads(full.read_text(encoding="utf-8"))
images = sorted(coco["images"], key=lambda im: im["id"])[:args.n]
ids = {im["id"] for im in images}
for i, im in enumerate(images, 1):
    path = OUT / "images" / im["file_name"]
    if not path.exists():
        fetch(IMG_URL.format(im["id"]), path)
    if i % 100 == 0:
        print(f"  이미지 {i}/{len(images)}")
subset = {"images": images, "categories": coco["categories"],
          "annotations": [a for a in coco["annotations"] if a["image_id"] in ids]}
(OUT / "instances.json").write_text(json.dumps(subset), encoding="utf-8")
print(f"[완료] {OUT}: 이미지 {len(images)}장, 정답 박스 {len(subset['annotations'])}개")
