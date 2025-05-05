import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import update_tracker
import time
import csv

# Load the YOLOv8 Medium model (better accuracy than 's' version)
model = YOLO('yolov8s.pt')

# Mouse callback function for pixel inspection (optional)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load video
cap = cv2.VideoCapture("veh2.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output writer
output = cv2.VideoWriter('output_tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (1020, 500))

# Load class names
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")


cy1 = 322
cy2 = 368
offset = 6
vh_down = {}
counter = []
vh_up = {}
counter1 = []

count = 0

with open('car_details.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Car ID", "Direction", "Speed (Km/h)", "Timestamp"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 3 != 1:
            continue

        frame = cv2.resize(frame, (1020, 500))

        # Improved: Lower confidence threshold
        results = model.predict(frame, conf=0.3)
        boxes = results[0].boxes.data
        px = pd.DataFrame(boxes).astype("float")
        detections = []

        for _, row in px.iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            class_id = int(row[5])
            class_name = class_list[class_id]
            if class_name in ['car', 'Van', 'truck', 'bus']:
                detections.append([x1, y1, x2, y2])

        bbox_id = update_tracker(detections)

        for x3, y3, x4, y4, obj_id in bbox_id:
            cx, cy = (x3 + x4) // 2, (y3 + y4) // 2
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

            # DOWN direction
            if cy1 - offset < cy < cy1 + offset:
                vh_down[obj_id] = time.time()
            if obj_id in vh_down and cy2 - offset < cy < cy2 + offset:
                elapsed = time.time() - vh_down[obj_id]
                if obj_id not in counter:
                    counter.append(obj_id)
                    speed_kmh = (10 / elapsed) * 3.6
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([obj_id, "Down", int(speed_kmh), timestamp])
                    cv2.putText(frame, f"{obj_id}", (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"{int(speed_kmh)} Km/h", (x4, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # UP direction
            if cy2 - offset < cy < cy2 + offset:
                vh_up[obj_id] = time.time()
            if obj_id in vh_up and cy1 - offset < cy < cy1 + offset:
                elapsed = time.time() - vh_up[obj_id]
                if obj_id not in counter1:
                    counter1.append(obj_id)
                    speed_kmh = (10 / elapsed) * 3.6
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([obj_id, "Up", int(speed_kmh), timestamp])
                    cv2.putText(frame, f"{obj_id}", (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"{int(speed_kmh)} Km/h", (x4, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Visuals
        cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
        cv2.putText(frame, 'L1', (277, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
        cv2.putText(frame, 'L2', (182, 367), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Count display
        cv2.putText(frame, f'Going down: {len(counter)}', (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f'Going up: {len(counter1)}', (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        output.write(frame)
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
output.release()
cv2.destroyAllWindows()

cap.release()
output.release()
cv2.destroyAllWindows()
