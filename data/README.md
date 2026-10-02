# data

실습용 이미지·영상은 저장소에 커밋하지 않고 스크립트로 내려받습니다.

```bash
python data/download_samples.py
```

- 내려받을 목록과 출처·라이선스: [samples.json](samples.json)
- 받은 파일은 `data/images/`, `data/videos/`에 저장됩니다 (`.gitignore` 대상)
- 공개 데이터셋(COCO 등)은 각 장 README에서 별도로 안내합니다
