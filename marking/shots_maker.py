import cv2
import os

# Константы для путей
ASSETS_DIR = "marking/assets"
VIDEOS_DIR = os.path.join(ASSETS_DIR, "videos")

def setup_folders():
    """Создает необходимые папки, если они не существуют"""
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
    if not os.path.exists(VIDEOS_DIR):
        os.makedirs(VIDEOS_DIR)

def get_next_shots_folder():
    """
    Находит следующую доступную папку shots в директории assets
    """
    index = 1
    while True:
        folder_name = f"shots{index}"
        full_path = os.path.join(ASSETS_DIR, folder_name)
        if not os.path.exists(full_path):
            return full_path
        index += 1

def extract_frames(video_path, output_folder, interval_sec=0.5):
    """
    Извлекает кадры из видео с заданным интервалом и сохраняет их в папку
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)
    if frame_interval < 1:
        frame_interval = 1
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count

def process_videos_folder(interval_sec=0.5):
    """
    Обрабатывает все видео файлы в папке assets/videos
    """
    setup_folders()
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
    video_files = [f for f in os.listdir(VIDEOS_DIR) 
                 if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"В папке {VIDEOS_DIR} нет видеофайлов!")
        print(f"Поддерживаемые форматы: {', '.join(video_extensions)}")
        return
    
    print(f"Найдено видеофайлов: {len(video_files)}")
    
    for video_file in video_files:
        # Создаем папку для этого видео
        shots_folder = get_next_shots_folder()
        os.makedirs(shots_folder)
        
        video_path = os.path.join(VIDEOS_DIR, video_file)
        print(f"\nОбработка видео: {video_file}")
        print(f"Создана папка для скриншотов: {shots_folder}")
        
        saved_count = extract_frames(video_path, shots_folder, interval_sec)
        print(f"Сохранено скриншотов: {saved_count}")

if __name__ == "__main__":
    # Интервал между скриншотами (в секундах)
    screenshot_interval = 2
    
    # Запускаем обработку
    print("=== Video Frame Extractor ===")
    print(f"Исходные видео: {VIDEOS_DIR}")
    print(f"Папка для результатов: {ASSETS_DIR}")
    
    process_videos_folder(screenshot_interval)
    print("\nОбработка всех видео завершена!")