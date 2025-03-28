from PIL import Image, ImageDraw
import os

image_path = "train/Dataset/obj_Train_data/frame_00000.jpg"
label_path = "train/Dataset/obj_Train_data/frame_00000.txt"

image = Image.open(image_path)
draw = ImageDraw.Draw(image)
w, h = image.size

with open(label_path, 'r') as f:
    for line in f:
        class_id, xc, yc, bw, bh = map(float, line.split())
        x1 = (xc - bw/2) * w
        y1 = (yc - bh/2) * h
        x2 = (xc + bw/2) * w
        y2 = (yc + bh/2) * h
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

image.show()