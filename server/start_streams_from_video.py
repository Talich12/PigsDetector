import subprocess
import json
import os
from threading import Thread
from pathlib import Path

# Конфигурация
HOME_DIR = str(Path.home())
MEDIAMTX_PATH = os.path.join(HOME_DIR, "mediamtx")
MEDIAMTX_CONFIG = os.path.join(HOME_DIR, "mediamtx.yml")
INPUT_FILES_JSON = "config.json"
RTSP_URL_TEMPLATE = "rtsp://localhost:8554/mystream{}"

def load_stream_files():
    """Загружает список видеофайлов из JSON с абсолютными путями"""
    try:
        with open(INPUT_FILES_JSON, 'r') as f:
            data = json.load(f)
            return data.get("stream_files", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ошибка загрузки {INPUT_FILES_JSON}: {str(e)}")
        return []

def run_mediamtx():
    """Запускает mediamtx сервер"""
    if not os.path.exists(MEDIAMTX_PATH):
        print(f"Ошибка: {MEDIAMTX_PATH} не найден!")
        return
    
    print(f"Запускаем mediamtx из {MEDIAMTX_PATH} с конфигом {MEDIAMTX_CONFIG}")
    subprocess.run([MEDIAMTX_PATH, MEDIAMTX_CONFIG])

def run_ffmpeg_stream(input_file, stream_name):
    """Запускает ffmpeg для стриминга одного файла"""
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден, пропускаем...")
        return
    
    stream_url = RTSP_URL_TEMPLATE.format(stream_name)
    cmd = [
        "ffmpeg",
        "-re",
        "-stream_loop", "-1",
        "-i", input_file,
        "-c", "copy",
        "-f", "rtsp",
        stream_url
    ]
    print(f"Запускаем стрим: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при запуске ffmpeg: {str(e)}")

def main():

    # Получаем список файлов для стриминга
    stream_files = load_stream_files()
    
    if not stream_files:
        print("Нет файлов для стриминга. Проверьте streams.json")
        return

    # Запускаем ffmpeg для каждого файла
    ffmpeg_threads = []
    for i, filename in enumerate(stream_files, 1):
        thread = Thread(target=run_ffmpeg_stream, args=(filename, i))
        thread.daemon = True
        thread.start()
        ffmpeg_threads.append(thread)
        print(f"Запущен стрим {filename} как {RTSP_URL_TEMPLATE.format(i)}")

    # Ждем завершения всех потоков
    for thread in ffmpeg_threads:
        thread.join()

    mtx_thread.join()

if __name__ == "__main__":
    main()