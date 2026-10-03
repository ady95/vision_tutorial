"""실습용 샘플 이미지·영상 내려받기

data/samples.json에 적힌 파일을 data/ 아래로 내려받는다.
이미 받은 파일은 건너뛴다.

실행:
    python data/download_samples.py
"""
import json
import shutil
import sys
import urllib.request
from pathlib import Path

# Windows 콘솔에서 한글이 깨지지 않도록 UTF-8로 출력
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent
MANIFEST = DATA_DIR / "samples.json"


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "vision-tutorial"})
    tmp = dest.with_suffix(dest.suffix + ".part")                   # 받다가 끊기면 반쪽 파일이 남지 않게
    with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)                            # 영상(약 43MB)도 메모리에 다 올리지 않고 저장
    tmp.replace(dest)


def main():
    samples = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failed = []
    for sample in samples:
        dest = DATA_DIR / sample["file"]
        if dest.exists():
            print(f"[건너뜀] {sample['file']}")
            continue
        try:
            download(sample["url"], dest)
            print(f"[완료]   {sample['file']} ({dest.stat().st_size // 1024} KB)")
        except Exception as e:
            failed.append(sample["file"])
            print(f"[실패]   {sample['file']}: {e}")
    if failed:
        raise SystemExit(f"{len(failed)}개 파일을 받지 못했습니다. 네트워크를 확인하고 다시 실행하세요.")


if __name__ == "__main__":
    main()
