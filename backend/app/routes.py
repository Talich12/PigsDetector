from flask import Flask, jsonify
import json
from app import app
import psutil
import os
import subprocess


process = None

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


@app.route('/start_script', methods=['GET'])
def start_script():
    global process
    if process is not None:
        return jsonify({'status': 'error', 'message': 'Script is already running'}), 400
    
    try:
        # Запуск скрипта в новом терминале (для Linux)
        process = subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', '/home/podonok/diplom/start_streams.sh; exec bash'])
        # Для macOS можно использовать: ['open', '-a', 'Terminal', './your_script.sh']
        # Для Windows: ['start', 'cmd', '/k', 'your_script.bat']
        
        return jsonify({'status': 'success', 'message': 'Script started successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stop_script', methods=['GET'])
def stop_script():
    global process
    if process is None:
        return jsonify({'status': 'error', 'message': 'No script is running'}), 400
    
    try:
        # Завершаем процесс
        process.terminate()
        process = None
        return jsonify({'status': 'success', 'message': 'Script stopped successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

