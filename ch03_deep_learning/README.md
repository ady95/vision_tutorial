# 03. Deep Learning Vision 시작하기

도서 페이지: https://wikidocs.net/book/21494 (목차의 03장)

## 설치

PyTorch를 GPU·CUDA에 맞게 설치합니다 (도서 00-2 참고).

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # NVIDIA GPU
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # GPU 없음
```

03-4·03-5는 처음 실행할 때 torchvision의 ResNet18 ImageNet 가중치(약 45MB)를 자동으로 내려받습니다.

## 실습 데이터

```bash
python data/make_parts.py                                              # 시험용 180장 (02장과 같음)
python data/make_parts.py --n 300 --seed 2027 --out data/parts_train   # 학습용 900장
```

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 03-1 | 왜 Deep Learning이 필요한가 | `03_1_neuron.py` | 실측 완료 |
| 03-2 | CNN 이해하기 | `03_2_convolution.py` | 실측 완료 |
| 03-3 | PyTorch로 이미지 분류하기 | `parts_cls.py` (공통 모듈), `03_3_classification.py` | 실측 완료 |
| 03-4 | Transfer Learning과 Data Augmentation | `03_4_transfer_learning.py` | 실측 완료 |
| 03-5 | 실습: 정상/불량 이미지 분류 | `03_5_defect_classification.py` | 실측 완료 |

실행은 저장소 루트에서 합니다. 예) `python ch03_deep_learning/03_3_classification.py`

## 실측 환경과 재현성

- RTX 3060 12GB, PyTorch 2.14.1+cu130 기준. 시드와 cuDNN 연산 방식을 고정해, 같은 GPU·소프트웨어에서는 출력이 완전히 같습니다
- GPU 종류가 다르거나 CPU로 실행하면 정확도 숫자가 조금 달라질 수 있습니다 (도서 03-3의 "GPU 없이 실행하면" 참고)
- 소요 시간 (RTX 3060): 03-3 약 1분, 03-4 약 5분 20초, 03-5 약 50초. 4코어 CPU에서는 03-3만 약 9분
- 03-5의 `--bench`는 저장된 모델(outputs/ch03/resnet18_parts.pt)로 추론 속도만 잽니다
