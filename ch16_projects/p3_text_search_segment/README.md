# 16-3 프로젝트: 자연어 이미지 검색·분할 시스템

CLIP ViT-L-14 + faiss로 문장에 맞는 사진을 찾고, 찾은 사진 안에서 대상을 Grounding DINO + SAM 2.1로 칠합니다.

## 준비

```bash
uv pip install -e ".[dl,foundation]"
python data/download_coco_val.py
```

## 실행 (저장소 루트에서)

```bash
python ch16_projects/p3_text_search_segment/search_segment.py                                              # 예시 질의 + 73개 클래스 평가
python ch16_projects/p3_text_search_segment/search_segment.py --query "a cat sleeping on a couch" --find cat   # 직접 질의
```

색인은 처음 실행할 때 `outputs/ch16/p3/coco500_vitl14.faiss`에 저장되고, 다음부터는 불러옵니다. RTX 3060에서 평가 전체가 약 4분 걸립니다.
