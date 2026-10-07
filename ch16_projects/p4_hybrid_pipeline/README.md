# 16-4 프로젝트: 산업 Vision Hybrid Pipeline

OpenCV 품질 검사 → YOLO(04-4) → Rule(02-6) → SAM → OCR → VLM(애매할 때만) → 리포트

## 준비

02-6, 04-4 실습을 먼저 마칩니다. VLM 서버(10장 README)와 OCR 엔진이 필요합니다.

```bash
uv pip install -e ".[dl,foundation,vlm]"
uv pip install rapidocr
```

## 실행 (저장소 루트에서)

실측은 RTX 3090 두 장에서 4B 서버를 GPU 0에, 파이프라인(YOLO·SAM)을 GPU 1에 두었습니다.

```bash
# 터미널 A (.venv-vllm) — 4B 서버를 GPU 0에
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.85

# 터미널 B (저장소 루트, .venv) — 파이프라인은 GPU 1에
CUDA_VISIBLE_DEVICES=1 python ch16_projects/p4_hybrid_pipeline/pipeline.py --vlm-url http://localhost:8000/v1
CUDA_VISIBLE_DEVICES=1 python ch16_projects/p4_hybrid_pipeline/pipeline.py --vlm-url http://localhost:8000/v1 --all-vlm   # 비교: 모든 사진을 VLM이
```

생산 라인 사진 150장과 정답은 처음 실행할 때 `data/parts_line/`에 만들어집니다. 리포트는 `outputs/ch16/p4/report.json`.
GPU 한 장에 4B 서버와 YOLO·SAM을 함께 올리는 구성은 실측하지 않았습니다. GPU가 한 장이면 VLM 서버를 다른 컴퓨터에 두거나 2B를 씁니다.
