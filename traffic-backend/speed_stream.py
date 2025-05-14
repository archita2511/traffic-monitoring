import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import update_tracker
from utils import estimate_speed, draw_annotations, draw_crossing_lines, draw_direction_count

YOLO_MODEL_PATH = "yolov8s.pt"
FRAME_SIZE = (1020, 500)
METERS_PER_PIXEL = 0.2
FRAME_SKIP = 3
LINE_Y_START = 322
LINE_Y_END = 368
LINE_TOLERANCE = 6
ALERT_SPEED_THRESHOLD = 20
SPEED_LIMIT_MIN = 10

model = YOLO(YOLO_MODEL_PATH)
frame_counter = 0
entry_down = {}
entry_up = {}
logged_down = set()
logged_up = set()

def is_within_band(cy, line_y):
    return line_y - LINE_TOLERANCE < cy < line_y + LINE_TOLERANCE

def classify_speed(speed):
    if speed > ALERT_SPEED_THRESHOLD:
        return "Overspeeding vehicle"
    elif speed >= SPEED_LIMIT_MIN:
        return "Within speed limit"
    else:
        return "Slow vehicle"

def generate_stream(video_path):
    global frame_counter
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    alert_banner = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter = frame_counter + 1
        if frame_counter < 10 or frame_counter % FRAME_SKIP != 0:
            continue

        resized_frame = cv2.resize(frame, FRAME_SIZE)
        results = model.predict(resized_frame)[0]
        detections_df = pd.DataFrame(results.boxes.data.cpu().numpy(), columns=["x1", "y1", "x2", "y2", "conf", "cls"])

        detections = []
        for _, row in detections_df.iterrows():
            if int(row.cls) in {2, 3, 5, 7}:
                detections.append([int(row.x1), int(row.y1), int(row.x2), int(row.y2)])

        tracked_objects = update_tracker(detections)

        for x1, y1, x2, y2, obj_id in tracked_objects:
            cx, cy = draw_annotations(resized_frame, (x1, y1, x2, y2), obj_id)

            if is_within_band(cy, LINE_Y_START):
                entry_down[obj_id] = frame_counter
            elif obj_id in entry_down and is_within_band(cy, LINE_Y_END) and obj_id not in logged_down:
                speed = estimate_speed(entry_down[obj_id], frame_counter, METERS_PER_PIXEL, fps)
                label = classify_speed(speed)
                draw_annotations(resized_frame, (x1, y1, x2, y2), obj_id, speed, label)
                logged_down.add(obj_id)
                if label == "Overspeeding vehicle":
                    alert_banner = 10

            if is_within_band(cy, LINE_Y_END):
                entry_up[obj_id] = frame_counter
            elif obj_id in entry_up and is_within_band(cy, LINE_Y_START) and obj_id not in logged_up:
                speed = estimate_speed(entry_up[obj_id], frame_counter, METERS_PER_PIXEL, fps)
                label = classify_speed(speed)
                draw_annotations(resized_frame, (x1, y1, x2, y2), obj_id, speed, label)
                logged_up.add(obj_id)
                if label == "Overspeeding vehicle":
                    alert_banner = 10

        draw_crossing_lines(resized_frame, LINE_Y_START, LINE_Y_END)
        draw_direction_count(resized_frame, len(logged_down), len(logged_up))

        if alert_banner > 0:
            h, w = resized_frame.shape[:2]
            cv2.rectangle(resized_frame, (0, 0), (w, 40), (0, 0, 255), -1)
            cv2.putText(resized_frame, "Over Speeding!", (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
            alert_banner -= 1

        _, buffer = cv2.imencode('.jpg', resized_frame)

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


    cap.release()