# 12. Vision 모델 경량화와 배포

도서 페이지: https://wikidocs.net/book/21494 (목차의 12장)

## 설치

`uv pip install -e ".[deploy]"` (TensorRT는 GPU·CUDA 버전에 맞춰 별도 설치)

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 12-2 | Vision 모델과 VLM Quantization | `12_2_quantization.py` | 예정 |
| 12-3 | VLM 서빙: Transformers, vLLM, llama.cpp | `12_3_vlm_serving.md` | 예정 |
| 12-4 | ONNX, TensorRT와 실시간 Edge Vision | `12_4_onnx_tensorrt.py` | 예정 |

실행은 저장소 루트에서 합니다. 예) `python ch12_deploy/12_2_quantization.py`
