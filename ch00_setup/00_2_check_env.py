"""00-2. 실습 환경 점검

설치된 주요 라이브러리 버전과 GPU 사용 가능 여부를 출력한다.
설치하지 않은 라이브러리는 버전이 "-"로 표시되며, 해당 장에 들어갈 때 설치하면 된다.

실행:
    python ch00_setup/00_2_check_env.py
"""
import importlib
import platform
import sys

# Windows 콘솔에서 한글이 깨지지 않도록 UTF-8로 출력
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# (import 이름, 표시 이름, 필요한 장)
PACKAGES = [
    ("numpy", "NumPy", "01~"),
    ("cv2", "OpenCV", "01~"),
    ("matplotlib", "Matplotlib", "01~"),
    ("torch", "PyTorch", "03~"),
    ("torchvision", "torchvision", "03~"),
    ("ultralytics", "Ultralytics", "04~"),
    ("transformers", "Transformers", "06~"),
    ("open_clip", "OpenCLIP", "07"),
    ("faiss", "FAISS", "07, 16"),
    ("rapidocr", "RapidOCR", "16"),
    ("openai", "OpenAI SDK", "10~"),
    ("vllm", "vLLM", "10~"),
    ("onnxruntime", "ONNX Runtime", "11"),
]


def package_version(module_name):
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "버전 정보 없음")


def check_gpu():
    try:
        import torch
    except Exception:
        print("GPU   : PyTorch 미설치 — 03장 전에 설치하세요")
        return
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(f"GPU {i} : {props.name} (전체 {total / 1024**3:.1f} GB, 사용 가능 {free / 1024**3:.1f} GB)")
            if free / total < 0.5:
                print(f"        다른 프로세스가 GPU {i} 메모리의 {1 - free / total:.0%}를 쓰고 있습니다 — nvidia-smi로 확인하세요")
        print(f"CUDA  : {torch.version.cuda}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("GPU   : Apple MPS 사용 가능")
    else:
        print("GPU   : 사용 불가 — CPU로 실행됩니다 (01~02장은 CPU로 충분)")


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS    : {platform.system()} {platform.release()}")
    print()
    print(f"{'Package':<14}{'Version':<16}{'Chapter'}")
    print("-" * 40)
    for module_name, label, chapters in PACKAGES:
        version = package_version(module_name)
        print(f"{label:<14}{version or '-':<16}{chapters}")
    print()
    check_gpu()


if __name__ == "__main__":
    main()
