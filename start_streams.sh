#!/bin/bash
# Файл для хранения PIDs
PID_FILE="/tmp/streams.pid"

# Очищаем предыдущий PID-файл
rm -f $PID_FILE

# Закрываем предыдущие процессы
pkill mediamtx 2>/dev/null
pkill ffmpeg 2>/dev/null
pkill -f "python3 multy_stream.py" 2>/dev/null

# Запускаем mediamtx
cd ~
./mediamtx mediamtx.yml &
echo $! >> $PID_FILE

# Переходим в директорию с Python-скриптами
cd /home/podonok/diplom/server/

# Запускаем streams.py
python3 start_streams_from_video.py &
echo $! >> $PID_FILE

# Даём время на инициализацию
sleep 3

# Запускаем multy_stream.py
python3 multy_stream.py &
echo $! >> $PID_FILE

sleep 15

STREAMS_COUNT=$(python3 -c "import json; f=open('/home/podonok/diplom/server/config.json'); data=json.load(f); print(len(data['stream_files']))")
RTSP_IP=$(python3 -c "import json; f=open('/home/podonok/diplom/server/config.json'); data=json.load(f); print(data['IP'])")

cd /home/podonok/diplom/server/streams
# Запускаем FFmpeg-стримы
for ((i=1; i<=STREAMS_COUNT; i++)); do
    ffmpeg \
        -rtsp_transport tcp \
        -i "rtsp://${RTSP_IP}:8554/yolo_output$i" \
        -c:v copy -c:a aac -fflags +genpts \
        -f hls -hls_time 1 -hls_list_size 0 -hls_flags delete_segments \
        -hls_segment_filename "output${i}_%03d.ts" \
        "output${i}.m3u8" &
    echo $! >> $PID_FILE
done

# Дополнительно сохраняем PIDS дочерних процессов
pgrep -P $(cat $PID_FILE | tr '\n' ',') >> $PID_FILE 2>/dev/null