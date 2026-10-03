"""Imagenette 내려받기 — ImageNet에서 10개 클래스만 골라 낸 작은 데이터셋 (06장)

fast.ai가 공개한 imagenette2-320(짧은 변 320px, 약 330MB)을 받아 data/datasets/imagenette2-320에 풉니다.
검증셋(val) 3,925장만 씁니다. 이미지는 ImageNet의 일부이므로 ImageNet 이용 약관(비상업 연구·교육)을 따릅니다.

실행:
    python data/download_imagenette.py
"""
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
DEST = Path(__file__).resolve().parent / "datasets"
OUT = DEST / "imagenette2-320"

if (OUT / "val").exists():
    print(f"[건너뜀] {OUT} 가 이미 있습니다")
    raise SystemExit
DEST.mkdir(parents=True, exist_ok=True)
tgz = DEST / "imagenette2-320.tgz"
print(f"내려받는 중: {URL}")
with urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "vision-tutorial"}), timeout=60) as r, open(tgz, "wb") as f:
    shutil.copyfileobj(r, f)
with tarfile.open(tgz) as tar:
    tar.extractall(DEST, filter="data")
tgz.unlink()
n = sum(1 for _ in (OUT / "val").rglob("*.JPEG"))
print(f"[완료] {OUT} (검증셋 {n}장)")
