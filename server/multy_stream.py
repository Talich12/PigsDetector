from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import random
import subprocess
from threading import Thread, Event
import queue
import json
import os
import time
from datetime import datetime
# Configuration
SHOW_DISPLAY = True  # Set to False to disable display windows
MODEL_PATH = "/home/podonok/diplom/best.pt"

# Load your trained model
model = YOLO(MODEL_PATH)

# Load streams from JSON
def load_streams_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
        return config.get('stream_files', [])

# Track data storage
class TrackData:
    def __init__(self):
        self.history = defaultdict(lambda: [])
        self.colors = defaultdict(tuple)
        self.weights = defaultdict(int)
        self.confidences = defaultdict(float)
        self.last_update_time = 0
        self.frame_data = []

def write_frame_data(frame_data, stream_idx):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"tracking_data_stream_{stream_idx+1}.json"
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "objects": frame_data
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved tracking data to {filename}")

def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def generate_random_weight():
    return random.randint(75, 90)

def process_stream(stream_idx, input_source, output_url, stop_event, frame_queue, conf, tracking_mode):
    track_data = TrackData()
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error opening stream {stream_idx+1}: {input_source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    print(f"Stream {stream_idx+1}: {input_source} -> {output_url}")
    print(f"Frame size: {width}x{height}, FPS: {fps}")

    ffmpeg_cmd = [
        'ffmpeg',
        '-re', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-vsync', 'cfr',
        '-f', 'rtsp',
        output_url
    ]

    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    
    # Переменная для отслеживания времени
    last_save_time = time.time()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"Stream {stream_idx+1} ended, reconnecting...")
                cap.release()
                cap = cv2.VideoCapture(input_source)
                if not cap.isOpened():
                    print(f"Failed to reconnect to stream {stream_idx+1}")
                    break
                continue
            
            results = model.track(frame, tracker="bytetrack.yaml", persist=True, conf=conf, verbose=False)
            annotated_frame = frame.copy()
            
            # Очищаем данные кадра
            track_data.frame_data = []
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.int().cpu().tolist()
                confidences = results[0].boxes.conf.float().cpu().tolist()
                
                for box, track_id, cls, confidence in zip(boxes, track_ids, clss, confidences):
                    x, y, w, h = box
                    center = (int(x), int(y))
                    
                    if track_id not in track_data.colors:
                        track_data.colors[track_id] = generate_random_color()
                        track_data.weights[track_id] = generate_random_weight()
                    
                    track_data.confidences[track_id] = confidence
                    
                    # Сохраняем данные объекта для JSON
                    obj_data = {
                        "id": int(track_id),
                        "bbox": {
                            "x": float(x),
                            "y": float(y),
                            "width": float(w),
                            "height": float(h)
                        },
                        "confidence": float(confidence),
                        "class": int(cls),
                        "timestamp": datetime.now().isoformat()
                    }
                    track_data.frame_data.append(obj_data)
                    
                    track_data.history[track_id].append(center)
                    if len(track_data.history[track_id]) > 30:
                        track_data.history[track_id].pop(0)
                    
                    if tracking_mode and len(track_data.history[track_id]) > 1:
                        points = np.array(track_data.history[track_id], dtype=np.int32)
                        cv2.polylines(annotated_frame, [points], isClosed=False, 
                                     color=track_data.colors[track_id], thickness=2)
                    
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                 track_data.colors[track_id], 2)
                    
                    confidence_percent = confidence * 100
                    label = f"ID: {track_id} Pig | {track_data.weights[track_id]}kg | {confidence_percent:.1f}%"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                track_data.colors[track_id], 2)
            
            # Сохраняем данные каждую секунду
            current_time = time.time()
            if current_time - last_save_time >= 1.0 and track_data.frame_data:
                write_frame_data(track_data.frame_data, stream_idx)
                last_save_time = current_time
            
            if annotated_frame.shape[:2] != (height, width):
                annotated_frame = cv2.resize(annotated_frame, (width, height))
            ffmpeg_process.stdin.write(annotated_frame.tobytes())
            
            if SHOW_DISPLAY:
                frame_queue.put((stream_idx, annotated_frame))

    finally:
        # Сохраняем последние данные перед выходом
        if track_data.frame_data:
            write_frame_data(track_data.frame_data, stream_idx)
        
        cap.release()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        print(f"Stream {stream_idx+1} processing stopped")

def display_frames(frame_queue, num_streams, stop_event):
    for i in range(num_streams):
        cv2.namedWindow(f'Stream {i+1}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Stream {i+1}', 640, 480)
    
    frames = {}
    last_active_time = time.time()
    
    while not stop_event.is_set():
        try:
            stream_idx, frame = frame_queue.get(timeout=1.0)
            frames[stream_idx] = frame
            last_active_time = time.time()
            
            for i, frame in frames.items():
                cv2.imshow(f'Stream {i+1}', frame)
            
            if time.time() - last_active_time > 5.0:
                print("No frames received for 5 seconds, checking streams...")
                last_active_time = time.time()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
                
        except queue.Empty:
            continue
    
    cv2.destroyAllWindows()

def main():
    streams = load_streams_config()
    f=open('/home/podonok/diplom/server/config.json')
    data=json.load(f)
    print(data['IP'])
    input_sources = [f"rtsp://localhost:8554/mystream{i+1}" for i in range(len(streams))]
    output_urls = [f"rtsp://{data['IP']}:8554/yolo_output{i+1}" for i in range(len(streams))]
    output_hls = [f"http://{data['IP']}:8000/output{i+1}.m3u8" for i in range(len(streams))]

    output_data = {'streams': output_hls}
    with open('streams.json', 'w') as f:
        json.dump(output_data, f, indent=4)

    stop_event = Event()
    frame_queue = queue.Queue(maxsize=10) if SHOW_DISPLAY else None
    threads = []

    try:
        if SHOW_DISPLAY:
            display_thread = Thread(target=display_frames, args=(frame_queue, len(streams), stop_event))
            display_thread.start()
        
        for i, (input_src, output_url) in enumerate(zip(input_sources, output_urls)):
            thread = Thread(target=process_stream, 
                          args=(i, input_src, output_url, stop_event, frame_queue, data['conf'], data['tracking_mode']))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            print(f"Started processing for stream {i+1}")

        if SHOW_DISPLAY:
            display_thread.join()
        else:
            while not stop_event.is_set():
                time.sleep(1)
        
    except KeyboardInterrupt:
        print("\nStopping all streams...")
        stop_event.set()
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=1.0)
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()
        print("All streams stopped")

if __name__ == "__main__":
    main()