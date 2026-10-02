# Vision 알고리즘 따라하기 — 예제 코드

위키독스 도서 **「Vision 알고리즘 따라하기 — OpenCV부터 YOLO·SAM·VLM까지」**의 예제 코드 저장소입니다.

- 도서: https://wikidocs.net/book/21494
- 책의 각 페이지 번호와 이 저장소의 폴더·파일 번호가 같습니다. 예) 02-2 페이지 → `ch02_rule_based/02_2_*.py`

> 이 저장소는 집필과 함께 채워지고 있습니다. 아직 비어 있는 장은 각 폴더의 README.md에 예정 목록만 있습니다.

## 폴더 구성

| 폴더 | 장 | 주요 기술 |
|---|---|---|
| [ch00_setup](ch00_setup) | 00. 시작하기 | 환경 점검, 빠른 시작 |
| [ch01_basics](ch01_basics) | 01. Computer Vision의 기초 | NumPy, OpenCV, Color Space |
| [ch02_rule_based](ch02_rule_based) | 02. Rule 기반 Vision | Threshold, Morphology, Contour, Matching |
| [ch03_deep_learning](ch03_deep_learning) | 03. Deep Learning Vision 시작하기 | PyTorch, CNN, Transfer Learning |
| [ch04_detection](ch04_detection) | 04. Object Detection과 YOLO | Ultralytics YOLO |
| [ch05_seg_pose_track](ch05_seg_pose_track) | 05. Segmentation · Pose · Tracking | YOLO Seg/Pose, ByteTrack |
| [ch06_vit](ch06_vit) | 06. Vision Transformer | ViT, DETR |
| [ch07_foundation](ch07_foundation) | 07. Vision Foundation Model | CLIP, DINOv2, Florence-2 |
| [ch08_sam](ch08_sam) | 08. Segment Anything | SAM 2, SAM 3 |
| [ch09_open_vocab](ch09_open_vocab) | 09. Open-Vocabulary Vision | Grounding DINO |
| [ch10_vlm](ch10_vlm) | 10. Vision Language Model | Qwen3.5 (2B·4B·9B), vLLM |
| [ch11_deploy](ch11_deploy) | 11. Vision 모델 경량화와 배포 | Quantization, vLLM, ONNX, TensorRT |
| [ch12_evaluation](ch12_evaluation) | 12. Vision 모델 평가 | 지표, VLM 평가, Benchmark |
| [ch13_selection](ch13_selection) | 13. 어떤 Vision 알고리즘을 선택해야 하는가 | 비교 실험 |
| [ch14_hybrid](ch14_hybrid) | 14. Hybrid Vision System | Cascade, Routing |
| [ch15_agent](ch15_agent) | 15. Vision Agent | VLM Tool Calling |
| [ch16_projects](ch16_projects) | 16. 실전 프로젝트 | 종합 프로젝트 4종 |
| [data](data) | 공통 | 샘플 이미지·영상 내려받기 |

> OCR은 이 책에서 따로 다루지 않습니다. 문자 인식과 Document AI는 「OCR 따라하기」([도서](https://wikidocs.net/book/21475), [예제 코드](https://github.com/ady95/ocr_tutorial))를 참고하세요.

## 시작하기

### 1. 저장소 받기

```bash
git clone https://github.com/ady95/vision_tutorial.git
cd vision_tutorial
```

### 2. 가상환경과 기본 패키지

[uv](https://docs.astral.sh/uv/) 사용을 권장합니다.

```bash
uv venv --python 3.11
uv pip install -e .            # 01~02장: NumPy, OpenCV, Matplotlib
```

필요한 장에 맞춰 추가 그룹을 설치합니다.

| 그룹 | 대상 장 | 설치 |
|---|---|---|
| `dl` | 03~06 | `uv pip install -e ".[dl]"` |
| `foundation` | 06~09 | `uv pip install -e ".[foundation]"` |
| `ocr` | 16 (번호판·Hybrid 프로젝트) | `uv pip install -e ".[ocr]"` |
| `vlm` | 10, 12, 14~16 | `uv pip install -e ".[vlm]"` |
| `deploy` | 11 | `uv pip install -e ".[deploy]"` |

PyTorch는 GPU·CUDA 버전에 맞는 빌드를 먼저 설치하는 편이 안전합니다. 설치 명령은 [PyTorch 공식 안내](https://pytorch.org/get-started/locally/)를 따르세요.

### 3. 환경 점검과 샘플 데이터

```bash
python ch00_setup/00_2_check_env.py
python data/download_samples.py
```

## 실행 환경

- Python 3.11 (3.10 이상)
- NVIDIA GPU: RTX 3060 12GB 기준으로 검증합니다. 24GB가 필요한 예제(Qwen3.5-9B 등)는 RTX 3090 24GB에서 검증했으며, 해당 README에 표시하고 12GB 대안을 함께 적습니다
- vLLM 예제(10장~)는 Linux + NVIDIA GPU 환경을 기준으로 합니다
- GPU가 없다면 01~02장은 CPU로 충분하고, 나머지는 Google Colab에서 실행할 수 있습니다

## 라이선스

- 이 저장소의 예제 코드: [MIT License](LICENSE)
- 예제가 사용하는 라이브러리와 모델 가중치는 각자의 라이선스를 따릅니다. 예) Ultralytics YOLO는 AGPL-3.0, SAM·Qwen 등은 각 배포처의 라이선스를 확인하세요. 장별 README에 해당 장에서 쓰는 라이선스를 적어 둡니다.
- 샘플 이미지·영상의 출처와 라이선스는 [data/samples.json](data/samples.json)에 기록합니다.
