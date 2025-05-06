import cv2
from ultralytics import YOLO
from tracker import Tracker
import pandas as pd

def generate_stream(path):
    model = YOLO("yolov8s.pt")
    cap = cv2.VideoCapture(path)
    tracker = Tracker(max_distance=40)

    meters_per_pixel = 0.2
    cy1, cy2 = 322, 368
    offset = 10
    vh_down, vh_up, counter, counter1 = {}, {}, [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0

    with open("coco.txt", "r") as f:
        class_list = f.read().splitlines()

    # To persist banner for a few frames
    banner_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 2 != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        boxes = results[0].boxes.data
        px = pd.DataFrame(boxes).astype("float")

        detections = []
        for _, row in px.iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            class_id = int(row[5])
            if 'car' in class_list[class_id]:
                detections.append([x1, y1, x2, y2])

        bbox_id = tracker.update(detections)
        for bbox in bbox_id:
            x3, y3, x4, y4, obj_id = bbox
            cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

            # Downward direction
            if cy1 - offset < cy < cy1 + offset:
                vh_down[obj_id] = count

            if obj_id in vh_down and cy2 - offset < cy < cy2 + offset:
                pixel_distance = abs(cy2 - cy1)
                real_world_distance = meters_per_pixel * pixel_distance
                frames_crossed = count - vh_down[obj_id]
                elapsed_time = (frames_crossed * 2) / fps

                if obj_id not in counter:
                    counter.append(obj_id)
                    speed_kmh = (real_world_distance / elapsed_time) * 3.6
                    if speed_kmh > 40:
                        label = "Overspeeding vehicle"
                    elif speed_kmh >= 20:
                        label = "Within speed limit"
                    else:
                        label = "Slow vehicle"
                    cv2.putText(frame, label, (x4, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Upward direction
            if cy2 - offset < cy < cy2 + offset:
                vh_up[obj_id] = count

            if obj_id in vh_up and cy1 - offset < cy < cy1 + offset:
                pixel_distance = abs(cy2 - cy1)
                real_world_distance = meters_per_pixel * pixel_distance
                frames_crossed = count - vh_up[obj_id]
                elapsed_time = (frames_crossed * 2) / fps

                if obj_id not in counter1:
                    counter1.append(obj_id)
                    speed_kmh = (real_world_distance / elapsed_time) * 3.6
                    label = f"{int(speed_kmh)} Km/h"

                    speed_limit = 40
                    if speed_kmh > speed_limit:
                        banner_frames = 30  # Show banner for next 30 frames

                    cv2.putText(frame, label, (x4, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show top banner if speeding detected in recent frames
        if banner_frames > 0:
            cv2.rectangle(frame, (0, 0), (1020, 40), (0, 0, 255), -1)
            cv2.putText(frame, "Over Speeding!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
            banner_frames -= 1

        # Draw lines
        cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
        cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
