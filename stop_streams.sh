#!/bin/bash
PID_FILE="/tmp/streams.pid"

# Если файла нет - просто выходим
if [ ! -f "$PID_FILE" ]; then
    echo "PID файл не найден, возможно стримы не запущены"
    exit 0
fi

# Останавливаем все процессы из файла
while read pid; do
    if ps -p $pid > /dev/null; then
        kill -9 $pid 2>/dev/null
    fi
done < "$PID_FILE"

# Дополнительная очистка
pkill -f ffmpeg 2>/dev/null
pkill -f mediamtx 2>/dev/null
pkill -f "python3 multy_stream.py" 2>/dev/null

# Удаляем PID-файл
rm -f "$PID_FILE"
echo "Все стримы остановлены"