import cv2
from ultralytics import YOLO
from tracker import update_tracker
import pandas as pd

# Load YOLO model once
model = YOLO("yolov8s.pt")

# Calibration: meters per pixel and frame skipping
METERS_PER_PIXEL = 0.2
FRAME_SKIP = 3

# Crossing lines (y‑coordinates) and tolerance
CY_START, CY_END = 322, 368
OFFSET = 6

# State for each direction
frame_count = 0
entries_down = {}
entries_up   = {}
counted_down = set()
counted_up   = set()

def generate_stream(path):
    global frame_count
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count < 10 or frame_count % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)[0]
        boxes = results.boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,cls]
        df = pd.DataFrame(boxes, columns=["x1","y1","x2","y2","conf","cls"])

        # Filter vehicle classes: car(2), motorcycle(3), bus(5), truck(7)
        detections = []
        for _, row in df.iterrows():
            if int(row.cls) in {2, 3, 5, 7}:
                x1, y1, x2, y2 = map(int, row[["x1","y1","x2","y2"]])
                detections.append([x1, y1, x2, y2])

        # Assign persistent IDs
        tracked = update_tracker(detections)

        for x1, y1, x2, y2, obj_id in tracked:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Downward direction timing
            if CY_START - OFFSET < cy < CY_START + OFFSET:
                entries_down[obj_id] = frame_count
            if (obj_id in entries_down and
                CY_END - OFFSET < cy < CY_END + OFFSET and
                obj_id not in counted_down):
                elapsed = (frame_count - entries_down[obj_id]) * FRAME_SKIP / fps
                pixel_dist = abs(CY_END - CY_START)
                speed = (METERS_PER_PIXEL * pixel_dist / elapsed) * 3.6
                counted_down.add(obj_id)
                cv2.putText(frame, f"{int(speed)} km/h",
                            (x2, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Upward direction timing
            if CY_END - OFFSET < cy < CY_END + OFFSET:
                entries_up[obj_id] = frame_count
            if (obj_id in entries_up and
                CY_START - OFFSET < cy < CY_START + OFFSET and
                obj_id not in counted_up):
                elapsed = (frame_count - entries_up[obj_id]) * FRAME_SKIP / fps
                pixel_dist = abs(CY_END - CY_START)
                speed = (METERS_PER_PIXEL * pixel_dist / elapsed) * 3.6
                counted_up.add(obj_id)
                cv2.putText(frame, f"{int(speed)} km/h",
                            (x2, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw reference lines
        h, w = frame.shape[:2]
        cv2.line(frame, (0, CY_START), (w, CY_START), (255, 255, 255), 1)
        cv2.line(frame, (0, CY_END),   (w, CY_END),   (255, 255, 255), 1)

        # Yield encoded JPEG frame
        ret, jpg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')

    cap.release()