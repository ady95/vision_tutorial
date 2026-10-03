"""01-2. Webcam에서 프레임 하나 읽기 (카메라가 연결된 PC에서 실행)

실행:
    python ch01_basics/01_2_webcam.py
"""
import cv2

cap = cv2.VideoCapture(0)          # 0번 카메라
ok, frame = cap.read()
print(ok, None if frame is None else frame.shape)
cap.release()
