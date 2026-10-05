# 15. Vision Agent

도서 페이지: https://wikidocs.net/book/21494 (목차의 15장)

15-1, 15-3은 개념 절이라 예제가 없습니다.

## 설치

```bash
uv pip install -e ".[dl,vlm]"
uv pip install rapidocr          # 15-2의 OCR 도구 (ONNX Runtime 기반)
```

vLLM 서버는 **도구 호출을 켜서** 띄웁니다. Qwen3.5의 도구 호출 형식은 `qwen3_coder` 파서입니다.

```bash
vllm serve Qwen/Qwen3.5-9B --port 8000 --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

15-2는 12-3의 평가셋(`data/parts_bench`)과 02-6의 기준값을 씁니다. 15-4는 05-5의 교통 영상(`data/download_samples.py`)을 씁니다.

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 15-2 | VLM + Tool Calling (YOLO·Rule 측정·OCR 도구) | `15_2_vlm_tool_calling.py` | 실측 완료 |
| 15-4 | 보고 판단하고 행동하는 Agent (갓길 감시, 안전 장치) | `15_4_vision_agent.py` | 실측 완료 |

실행은 저장소 루트에서 합니다. 예) `python ch15_agent/15_4_vision_agent.py`

## 소요 시간 (RTX 3090)

- 15-2: 약 20분 (부품 420장 x 2, COCO 100장 x 2)
- 15-4: 약 2분
