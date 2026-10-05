# 11. Vision 모델 경량화와 배포

도서 페이지: https://wikidocs.net/book/21494 (목차의 11장)

## 설치

```bash
uv pip install -e ".[dl,foundation,vlm,deploy]"   # onnx, onnxslim, onnxruntime 포함
uv pip install tensorrt-cu13                       # NVIDIA GPU, PyTorch가 CUDA 13 빌드일 때 (11-2·11-4)
```

- onnxruntime-gpu 1.30은 CUDA 12 라이브러리(libcublasLt.so.12)를 찾으므로 CUDA 13 PyTorch 환경에서는 GPU 실행 경로가 동작하지 않습니다. 이 장은 ONNX Runtime을 CPU에서, GPU에서는 TensorRT를 씁니다
- vLLM 서버는 10장 README와 같습니다. 양자화 체크포인트: `RedHatAI/Qwen3.5-9B-quantized.w4a16`, `RedHatAI/Qwen3.5-9B-quantized.w8a8`

### llama.cpp (11-2·11-3)

공식 릴리스의 리눅스 실행 파일은 Ubuntu 24.04(glibc 2.38) 기준이라 Ubuntu 22.04에서는 소스로 빌드합니다 (CUDA 툴킷 필요).

```bash
git clone --depth 1 --branch b11405 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF   # RTX 30 시리즈 = 86
cmake --build build -j --target llama-server llama-mtmd-cli

hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf mmproj-F16.gguf --local-dir models/qwen35-9b
./build/bin/llama-server -m models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf --mmproj models/qwen35-9b/mmproj-F16.gguf \
  -ngl 99 -c 16384 -np 8 --port 8000 --jinja
```

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 11-1 | GPU 메모리와 정밀도 (FP32/FP16/BF16, YOLO Activation, Qwen3.5 KV Cache) | `11_1_memory_precision.py` | 실측 완료 |
| 11-2 | YOLO26s FP16·INT8 (TensorRT) 정확도 | `11_2_vision_int8.py` | 실측 완료 |
| 11-2 | Qwen3.5-9B BF16·W8A8·W4A16·Q4_K_M 품질 비교 | `11_2_vlm_quant.py` | 실측 완료 |
| 11-3 | Transformers · vLLM · llama.cpp 서빙 비교 | `11_3_serving_bench.py` | 실측 완료 |
| 11-4 | ONNX Runtime · TensorRT Latency / Throughput / FPS | `11_4_onnx_tensorrt.py` | 실측 완료 |

실행은 저장소 루트에서 합니다. 예) `python ch11_deploy/11_1_memory_precision.py`

## 소요 시간

- 11-2 YOLO: TensorRT 엔진 빌드 2~4분씩 + 평가 약 3분 (RTX 3060)
- 11-2 VLM 시험 세트: 모델마다 약 20~30분 (좌표 388문항이 대부분)
- 11-4: 약 5분 (배치 16 엔진 빌드 포함)
