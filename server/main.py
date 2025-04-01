import cv2
from ultralytics import YOLO
import subprocess
import numpy as np

# 1. Загружаем модель YOLO
model = YOLO("best.pt")  # или ваша best.pt

# 2. Источник видео (веб-камера)
input_source = 0  # 0 = веб-камера

# 3. Получаем параметры видео потока
cap = cv2.VideoCapture(input_source)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

print(f"Размер кадра: {width}x{height}, FPS: {fps}")

# 4. Настройки FFmpeg для RTSP
rtsp_output_url = "rtsp://192.168.0.105:8554/yolo_output"
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{width}x{height}',
    '-r', str(fps),
    '-i', '-',
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-f', 'rtsp',
    rtsp_output_url
]

# 5. Запускаем FFmpeg
ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera Preview", frame)
        # Обработка YOLO
        results = model(frame)
        annotated_frame = results[0].plot()

        # Проверка размера кадра
        if annotated_frame.shape[0] != height or annotated_frame.shape[1] != width:
            annotated_frame = cv2.resize(annotated_frame, (width, height))

        # Отправка кадра в FFmpeg
        ffmpeg_process.stdin.write(annotated_frame.tobytes())

        # Превью (опционально)
        cv2.imshow("YOLO Output", annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # Очистка ресурсов
    cap.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    cv2.destroyAllWindows()