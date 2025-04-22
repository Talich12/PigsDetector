from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import random

# Загрузка вашей обученной модели
model = YOLO("/home/podonok/diplom/best.pt")

# Открытие видеофайла или камеры
video_path = "/home/podonok/diplom/tracking/1.mp4"
cap = cv2.VideoCapture(video_path)

# Проверка, открылось ли видео
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видеофайл.")
    exit()

# Словари для хранения данных о треках
track_history = defaultdict(lambda: [])
track_colors = defaultdict(tuple)
track_weights = defaultdict(int)  # Словарь для хранения весов

# Генерация случайных цветов для треков
def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Генерация случайного веса свиньи (75-90 кг)
def generate_random_weight():
    return random.randint(75, 90)

# Цикл обработки видео
while True:
    # Чтение кадра
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera Preview", frame)
    # Трекинг объектов на кадре
    results = model.track(frame, tracker="bytetrack.yaml", persist=True, conf=0.5)
    
    # Если есть обнаруженные объекты с треками
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist()
        
        for box, track_id, cls in zip(boxes, track_ids, clss):
            x, y, w, h = box
            center = (int(x), int(y))
            
            # Если это новый трек, генерируем для него цвет и вес
            if track_id not in track_colors:
                track_colors[track_id] = generate_random_color()
                track_weights[track_id] = generate_random_weight()
            
            # Сохраняем центр объекта в историю
            track_history[track_id].append(center)
            
            # Ограничиваем длину истории
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)
            
            # Рисуем историю пути
            if len(track_history[track_id]) > 1:
                points = np.array(track_history[track_id], dtype=np.int32)
                cv2.polylines(frame, [points], isClosed=False, color=track_colors[track_id], thickness=2)
            
            # Рисуем bounding box с цветом трека
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), track_colors[track_id], 2)
            
            # Подпись с ID, классом и весом
            label = f"ID: {track_id} Pig | {track_weights[track_id]}kg"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_colors[track_id], 2)
    
    # Отображение кадра
    cv2.imshow("Pig Tracking with Weights", frame)
    
    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()