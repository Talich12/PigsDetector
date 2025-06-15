from flask import Flask, jsonify, request
import json
from app import app
import psutil
import os
import subprocess
import time
import atexit


process = None
CONFIG_PATH = "/home/podonok/diplom/server/config.json"
DATA_FOLDER = "/home/podonok/diplom/server/"

def start_streams_script():
    """Запускает скрипт start_streams.sh в фоне"""
    os.system("nohup /bin/bash start_streams.sh > /tmp/streams.log 2>&1 &")

def stop_streams_script():
    """Запускает скрипт stop_streams.sh и возвращает результат"""
    return os.popen("/bin/bash stop_streams.sh").read().strip()


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
    
    stop_streams_script()
    start_streams_script()

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
    
    stop_streams_script()
    start_streams_script()

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