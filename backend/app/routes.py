from flask import Flask, jsonify, request
import json
from app import app
import psutil
import os
import subprocess
import time
import atexit
import shutil
from glob import glob
from natsort import natsorted
from datetime import datetime


process = None
CONFIG_PATH = "/home/podonok/diplom/server/config.json"
DATA_FOLDER = "/home/podonok/diplom/server/"

def combine_ts_streams(input_dir="/home/podonok/diplom/server/streams", output_dir="/home/podonok/diplom/server/videos"):
    """
    Объединяет .ts файлы для каждого стрима и очищает исходную директорию
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Находим уникальные имена стримов
    ts_files = glob(os.path.join(input_dir, "*.ts"))
    stream_names = {os.path.basename(f).split("_")[0] for f in ts_files}
    
    if not stream_names:
        print("Нет .ts файлов для обработки")
        return

    print(f"Найдены стримы: {', '.join(stream_names)}")

    # Временная папка для резервного копирования (на случай ошибки)
    temp_backup_dir = os.path.join(input_dir, "temp_backup")
    os.makedirs(temp_backup_dir, exist_ok=True)

    try:
        for stream in sorted(stream_names):
            stream_files = natsorted(glob(os.path.join(input_dir, f"{stream}_*.ts")))
            
            if not stream_files:
                print(f"Пропуск стрима {stream} - нет файлов")
                continue

            # Создаём резервную копию перед обработкой
            for file in stream_files:
                shutil.copy2(file, temp_backup_dir)

            # Генерируем список для ffmpeg
            list_file = os.path.join(input_dir, f"{stream}_list.txt")
            with open(list_file, "w") as f:
                for file in stream_files:
                    f.write(f"file '{os.path.basename(file)}'\n")

            # Выходной файл с timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file = os.path.join(output_dir, f"{stream}_{timestamp}.mp4")

            print(f"Обработка {stream} ({len(stream_files)} файлов)...")

            try:
                subprocess.run([
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file,
                    "-c", "copy",
                    "-y",
                    output_file
                ], check=True)
                
                # Удаляем только если конвертация успешна
                for file in stream_files:
                    os.unlink(file)
                print(f"Видео сохранено: {output_file}")

            except subprocess.CalledProcessError as e:
                print(f"Ошибка конвертации {stream}: {e}")
                print("Восстанавливаем файлы из резервной копии")
                for file in stream_files:
                    if not os.path.exists(file):
                        backup = os.path.join(temp_backup_dir, os.path.basename(file))
                        shutil.copy2(backup, file)
            finally:
                if os.path.exists(list_file):
                    os.unlink(list_file)

    finally:
        # Удаляем резервную копию в любом случае
        if os.path.exists(temp_backup_dir):
            shutil.rmtree(temp_backup_dir)
        
        # Дополнительная очистка пустой директории
        if not glob(os.path.join(input_dir, "*")):
            print(f"Директория {input_dir} пуста")
        else:
            print("Остались необработанные файлы")


def start_streams_script():
    """Запускает скрипт start_streams.sh в фоне"""
    os.system("nohup /bin/bash start_streams.sh > /tmp/streams.log 2>&1 &")

def stop_streams_script():
    """Запускает скрипт stop_streams.sh и возвращает результат"""
    return os.popen("/bin/bash stop_streams.sh").read().strip()

def refresh_streams():
    combine_ts_streams()
    start_streams_script()
    stop_streams_script()

@app.route('/start')
def start_streams():
    try:
        start_streams_script()
        return "Стримы запускаются... Проверьте логи в /tmp/streams.log"
    except Exception as e:
        return f"Ошибка запуска: {str(e)}"

@app.route('/stop')
def stop_streams():
    try:
        result = stop_streams_script()
        return result or "Все стримы остановлены"
    except Exception as e:
        return f"Ошибка остановки: {str(e)}"


@app.route('/change_conf', methods=['POST'])
def change_conf():
    data = request.get_json()
    print(data)
    # Читаем текущий конфиг
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    print(config)
    # Обновляем нужные параметры
    config['conf'] = data['conf']
    
    # Записываем обратно с правильным форматированием
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    refresh_streams()

    return {"status": True}

@app.route('/change_tracking_mode', methods=['POST'])
def change_tracking_mode():
    data = request.get_json()
    print(data)
    # Читаем текущий конфиг
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    print(config)
    # Обновляем нужные параметры
    config['tracking_mode'] = data['tracking_mode']
    
    # Записываем обратно с правильным форматированием
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    refresh_streams()

    return {"status": True}

@app.route('/change_clahe', methods=['POST'])
def change_clahe():
    data = request.get_json()
    print(data)
    # Читаем текущий конфиг
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    print(config)
    # Обновляем нужные параметры
    config['clahe'] = data['clahe']
    
    # Записываем обратно с правильным форматированием
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    refresh_streams()

    return {"status": True}

@app.route('/change_model', methods=['POST'])
def change_model():
    data = request.get_json()
    print(data)
    # Читаем текущий конфиг
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    print(config)
    # Обновляем нужные параметры
    config['model'] = data['model']
    
    # Записываем обратно с правильным форматированием
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    refresh_streams()

    return {"status": True}

@app.route('/data/<id>')
def data(id):

    with open(DATA_FOLDER + f'tracking_data_stream_{id}.json', 'r') as f:
        data = json.load(f)

    return data

@app.route('/streams', methods=['GET'])
def send_json():
    with open('../server/streams.json', 'r') as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/stats', methods=['GET'])
def stats():
    import psutil

    # Загрузка CPU (%)
    cpu_usage = psutil.cpu_percent(interval=1)

    # Использование памяти
    memory = psutil.virtual_memory().percent
    data = {'cpu': cpu_usage,
            'memory': memory}
    
    return json.dumps(data)

def auto_start_streams():
    """Автоматически вызывает /start при запуске"""
    with app.test_request_context():
        response = start_streams()
        print(response)  # Для отладки

# Регистрируем остановку при завершении
atexit.register(lambda: stop_streams())

# Автозапуск только в основном процессе (не в reloader)
if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
    time.sleep(2)  # Даем время на инициализацию Flask
    auto_start_streams()