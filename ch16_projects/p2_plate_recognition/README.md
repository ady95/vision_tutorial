# 16-2 프로젝트: 차량 번호판 인식 시스템

YOLO26s + ByteTrack(05-5) → 번호판 찾기(A. Rule 04-6 / B. Grounding DINO) → 정면화 → RapidOCR(인식만) → 형식 규칙 → 추적 ID별 다수결 → 애매할 때만 VLM

## 준비

04-6, 05-5 실습을 먼저 마칩니다(교통 영상 `data/videos/traffic.mp4`와 04-6의 `find_plate`를 씁니다). VLM 서버(10장 README)와 OCR 엔진이 필요합니다.

```bash
uv pip install -e ".[dl,foundation,vlm]"
uv pip install rapidocr
```

## 실행 (저장소 루트에서)

```bash
python ch16_projects/p2_plate_recognition/plate_pipeline.py --step collect                       # 영상 전체에서 찾고 읽기 (약 30분)
python ch16_projects/p2_plate_recognition/plate_pipeline.py --step time                          # 단계별 시간
python ch16_projects/p2_plate_recognition/plate_pipeline.py --step candidates --vlm-url http://localhost:8000/v1   # 정답 만들기 도우미
python ch16_projects/p2_plate_recognition/plate_pipeline.py --step eval --vlm-url http://localhost:8000/v1
```

## 정답과 개인정보

번호판은 개인정보입니다. 정답은 저장소에 올리지 않습니다.

1. `--step candidates`가 차량마다 정답용 사진과 후보(VLM 답, OCR 다수결 상위 3개)를 `data/plate_gt/`에 만듭니다 (`.gitignore`에 포함)
2. 사진을 보고 `data/plate_gt/gt.json`에 `{"추적 ID": 맞는 후보 번호 | 직접 읽은 글자 | null}`을 적습니다. 사람도 읽을 수 없는 번호판은 null로 둡니다
3. `--step eval`은 정확도와 개수만 출력하고 번호판 글자는 출력하지 않습니다

결과를 공유할 때는 번호판이 보이는 사진과 정답 파일을 함께 올리지 않습니다.
