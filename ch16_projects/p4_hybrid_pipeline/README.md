# 16-4 프로젝트: 산업 Vision Hybrid Pipeline

OpenCV 품질 검사 → YOLO(04-4) → Rule(02-6) → SAM → OCR → VLM(애매할 때만) → 리포트

## 준비

02-6, 04-4 실습을 먼저 마칩니다. VLM 서버(10장 README)와 OCR 엔진이 필요합니다.

```bash
uv pip install -e ".[dl,foundation,vlm]"
uv pip install rapidocr
```

## 실행 (저장소 루트에서)

```bash
python ch16_projects/p4_hybrid_pipeline/pipeline.py --vlm-url http://localhost:8000/v1
python ch16_projects/p4_hybrid_pipeline/pipeline.py --vlm-url http://localhost:8000/v1 --all-vlm   # 비교: 모든 사진을 VLM이
```

생산 라인 사진 150장과 정답은 처음 실행할 때 `data/parts_line/`에 만들어집니다. 리포트는 `outputs/ch16/p4/report.json`.
12GB GPU 한 장에서는 4B 서버와 YOLO·SAM을 함께 올리기 어려우므로, VLM 서버를 다른 GPU에 두거나 2B를 씁니다.
