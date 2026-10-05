# 10. Vision Language Model

도서 페이지: https://wikidocs.net/book/21494 (목차의 10장)

## 설치

예제(클라이언트)는 저장소의 가상환경에 설치합니다.

```bash
uv pip install -e ".[vlm,dl,foundation]"
```

vLLM 서버는 PyTorch 버전을 고정해 쓰므로 Linux GPU 환경의 **별도 가상환경**에 설치합니다.

```bash
uv venv -p 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python vllm==0.30.0
source .venv-vllm/bin/activate                # 활성화하지 않으면 "ninja"를 찾지 못해 실패
export VLLM_USE_FLASHINFER_SAMPLER=0          # CUDA 툴킷(nvcc)이 없는 컴퓨터에서 필요
```

## 서버 실행 (검증한 설정)

| 모델 | GPU | 명령 |
|---|---|---|
| Qwen3.5-2B | RTX 3060 12GB | `vllm serve Qwen/Qwen3.5-2B --port 8000 --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.85` |
| Qwen3.5-4B | RTX 3060 12GB | 아래 참고 |
| Qwen3.5-9B | RTX 3090 24GB | `vllm serve Qwen/Qwen3.5-9B --port 8000 --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.92` |

4B는 12GB에 빠듯해서 이미지·영상 크기 상한과 메모리 조각화 설정이 필요합니다.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.86 \
  --limit-mm-per-prompt '{"image": {"count": 2, "width": 1280, "height": 1280}, "video": {"count": 1, "num_frames": 120, "width": 640, "height": 360}}'
```

- `--max-num-seqs 16`: 기본값(256)이면 9B가 3090 한 장에서도 메모리 부족
- 8000번 포트를 다른 프로그램이 쓰면 `--port 8001`로 띄우고 예제에 `--base-url http://localhost:8001/v1`을 줍니다
- 1920×1080 프레임은 예제에서 긴 변 1280으로 줄여 보냅니다(`ask(..., max_side=1280)`)

## 실습 데이터

```bash
python data/download_coco_val.py     # COCO val2017 앞 500장 (10-3·10-4·10-5·10-7)
python data/download_samples.py      # bus.jpg, coco_cats.jpg, 05-5의 교통 영상
```

10-5·10-6의 차량 사진(`--images`, `--same`, `--other`)은 저자 본인의 사진이라 배포하지 않습니다. 직접 찍은 사진으로 바꿔 실행합니다.

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| - | 공통 클라이언트 (ask, encode) | `vlm.py` | - |
| 10-2 | Qwen3.5 시작하기 (thinking 끄기·켜기, transformers 직접 실행) | `10_2_qwen35_hello.py` | 실측 완료 |
| 10-3 | 실습: Captioning과 Visual Question Answering | `10_3_caption_vqa.py` | 실측 완료 |
| 10-4 | 실습: Visual Grounding과 공간 이해 | `10_4_grounding_spatial.py` | 실측 완료 |
| 10-5 | 실습: 관계 이해와 Visual Reasoning | `10_5_reasoning.py` | 실측 완료 |
| 10-6 | 실습: Multi-image와 Video VLM | `10_6_video_vlm.py` | 실측 완료 |
| 10-7 | VLM의 Hallucination과 한계 (유도 질문, POPE) | `10_7_hallucination.py` | 실측 완료 |

실행은 저장소 루트에서, 서버를 띄운 뒤 합니다. 예) `python ch10_vlm/10_3_caption_vqa.py`

## 소요 시간

- 10-4 COCO 1,985문항: 2B 약 50분, 4B 약 89분(RTX 3060), 9B 약 76분(RTX 3090) (`--skip-coco`로 건너뜀)
- 10-7 POPE 4,410문항: 모델마다 수십 분
- 나머지: 각 1~3분
