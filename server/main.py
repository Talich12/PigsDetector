from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import random
import subprocess

# Load your trained model
model = YOLO("/home/podonok/diplom/best.pt")

# Input RTSP stream
input_source = "rtsp://localhost:8554/mystream"  # Replace with your input RTSP URL

# Get video stream parameters
cap = cv2.VideoCapture(input_source)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

print(f"Frame size: {width}x{height}, FPS: {fps}")

# RTSP output settings
rtsp_output_url = "rtsp://192.168.0.105:8554/yolo_output"  # Replace with your output RTSP URL
ffmpeg_cmd = [
    'ffmpeg',
    '-re',  # Важно! Режим реального времени
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{width}x{height}',
    '-r', '25',
    '-i', '-',
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-vsync', 'cfr',
    '-f', 'rtsp',
    rtsp_output_url
]

# Start FFmpeg process
ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# Dictionaries to store track data
track_history = defaultdict(lambda: [])
track_colors = defaultdict(tuple)
track_weights = defaultdict(int)

def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def generate_random_weight():
    return random.randint(75, 90)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Camera Preview", frame)
        # Object tracking on the frame
        results = model.track(frame, tracker="bytetrack.yaml", persist=True, conf=0.5, verbose=False)
        annotated_frame = frame.copy()
        
        # If there are detected objects with tracks
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, cls in zip(boxes, track_ids, clss):
                x, y, w, h = box
                center = (int(x), int(y))
                
                # If this is a new track, generate color and weight
                if track_id not in track_colors:
                    track_colors[track_id] = generate_random_color()
                    track_weights[track_id] = generate_random_weight()
                
                # Save object center to history
                track_history[track_id].append(center)
                
                # Limit history length
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)
                
                # Draw path history
                if len(track_history[track_id]) > 1:
                    points = np.array(track_history[track_id], dtype=np.int32)
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=track_colors[track_id], thickness=2)
                
                # Draw bounding box with track color
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), track_colors[track_id], 2)
                
                # Label with ID, class and weight
                label = f"ID: {track_id} Pig | {track_weights[track_id]}kg"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_colors[track_id], 2)
        
        # Check frame size and resize if needed
        if annotated_frame.shape[0] != height or annotated_frame.shape[1] != width:
            annotated_frame = cv2.resize(annotated_frame, (width, height))
        
        # Send frame to FFmpeg
        ffmpeg_process.stdin.write(annotated_frame.tobytes())
        
        # Optional preview
        cv2.imshow("Pig Tracking with Weights", annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # Cleanup
    cap.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    cv2.destroyAllWindows()