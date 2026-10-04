# 08. Segment Anything

도서 페이지: https://wikidocs.net/book/21494 (목차의 08장)

## 설치

```bash
uv pip install -e ".[dl,foundation]"
```

SAM(facebook/sam-vit-base), SAM 2.1(facebook/sam2.1-hiera-base-plus, -tiny), SAM 3(facebook/sam3)는 transformers에 내장된 구현을 씁니다.
SAM 3의 마스크 후처리(NMS·구멍 메우기)를 쓰려면 `kernels` 패키지가 필요합니다 (foundation 그룹에 포함).

### SAM 3 가중치 (08-4)

facebook/sam3는 Hugging Face에서 **접근 승인**을 받아야 내려받을 수 있습니다.

1. https://huggingface.co/facebook/sam3 에서 접근 요청 → 승인 메일 확인
2. 실행할 컴퓨터에서 `hf auth login` (토큰은 저장소나 코드에 넣지 마세요)

SAM 3는 Meta의 SAM License(상업적 사용 가능, 발표 시 사용 사실 표기, 군사 등 금지 용도)를 따릅니다. SAM·SAM 2는 Apache 2.0입니다.

## 실습 데이터

```bash
python data/download_coco_val.py     # 08-2·08-3·08-4: COCO val2017 앞 500장 + 정답 마스크
python data/download_samples.py      # 08-4: 05-5의 교통 영상
```

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 08-2 | 실습: Point·Box Prompt와 자동 마스크 생성 (COCO 정답 마스크 채점) | `08_2_sam_prompt.py` | 실측 완료 |
| 08-3 | 실습: Detection + SAM (YOLO26s-seg 마스크와 비교) | `08_3_yolo_sam.py` | 실측 완료 |
| 08-4 | SAM 3: 글·예시 Prompt, COCO 있는 것·없는 것 찾기, 영상에서 "red car" 추적 | `08_4_sam3.py` | 실측 완료 |

실행은 저장소 루트에서 합니다. 예) `python ch08_sam/08_3_yolo_sam.py --image 내_차량_사진.jpg`

## 소요 시간 (RTX 3060)

- 08-2: 세 모델 x COCO 500장 약 7분
- 08-3: 약 3분
- 08-4: COCO 500장 약 15분 + 영상 180프레임 약 2분 (영상은 bfloat16, GPU 메모리 약 3.4GB)
