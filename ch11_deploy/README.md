# 11. Vision 모델 경량화와 배포

도서 페이지: https://wikidocs.net/book/21494 (목차의 11장)

## 설치

`uv pip install -e ".[deploy]"` (TensorRT는 GPU·CUDA 버전에 맞춰 별도 설치)

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 11-2 | Vision 모델과 VLM Quantization | `11_2_quantization.py` | 예정 |
| 11-3 | VLM 서빙: Transformers, vLLM, llama.cpp | `11_3_vlm_serving.md` | 예정 |
| 11-4 | ONNX, TensorRT와 실시간 Edge Vision | `11_4_onnx_tensorrt.py` | 예정 |

실행은 저장소 루트에서 합니다. 예) `python ch11_deploy/11_2_quantization.py`
