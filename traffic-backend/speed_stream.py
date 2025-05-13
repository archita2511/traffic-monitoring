import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import update_tracker

# ------------------ Configuration ------------------

YOLO_MODEL_PATH = "yolov8s.pt"
FRAME_SIZE = (1020, 500)
METERS_PER_PIXEL = 0.2
FRAME_SKIP = 3
LINE_Y_START, LINE_Y_END = 322, 368
LINE_TOLERANCE = 6
ALERT_SPEED_THRESHOLD = 20
SPEED_LIMIT_MIN = 10

model = YOLO(YOLO_MODEL_PATH)
frame_counter = 0
entry_down, entry_up = {}, {}
logged_down, logged_up = set(), set()

def is_within_band(cy, line_y):
    return line_y - LINE_TOLERANCE < cy < line_y + LINE_TOLERANCE

def calculate_speed(start_frame, end_frame, fps):
    time_seconds = (end_frame - start_frame) * FRAME_SKIP / fps
    distance_m = abs(LINE_Y_END - LINE_Y_START) * METERS_PER_PIXEL
    return (distance_m / time_seconds) * 3.6

def classify_speed(speed):
    if speed > ALERT_SPEED_THRESHOLD:
        return "Overspeeding vehicle"
    elif speed >= SPEED_LIMIT_MIN:
        return "Within speed limit"
    else:
        return "Slow vehicle"

def draw_vehicle_info(frame, bbox, obj_id, label=None):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if label:
        cv2.putText(frame, label, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return cx, cy

def draw_lines_and_counts(frame, down_count, up_count, banner=False):
    h, w = frame.shape[:2]
    cv2.line(frame, (0, LINE_Y_START), (w, LINE_Y_START), (255, 255, 255), 1)
    cv2.line(frame, (0, LINE_Y_END),   (w, LINE_Y_END),   (255, 255, 255), 1)
    cv2.putText(frame, f'Going down: {down_count}', (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Going up: {up_count}',     (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    if banner:
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 255), -1)
        cv2.putText(frame, "Over Speeding!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def generate_stream(video_path):
    global frame_counter
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    alert_banner = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter < 10 or frame_counter % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, FRAME_SIZE)
        results = model.predict(frame)[0]
        detections_df = pd.DataFrame(results.boxes.data.cpu().numpy(), columns=["x1", "y1", "x2", "y2", "conf", "cls"])

        # Filter vehicles
        detections = []
        for _, row in detections_df.iterrows():
            if int(row.cls) in {2, 3, 5, 7}:
                detections.append([int(row.x1), int(row.y1), int(row.x2), int(row.y2)])

        tracked_objects = update_tracker(detections)

        for x1, y1, x2, y2, obj_id in tracked_objects:
            cx, cy = draw_vehicle_info(frame, (x1, y1, x2, y2), obj_id)

            if is_within_band(cy, LINE_Y_START):
                entry_down[obj_id] = frame_counter
            elif obj_id in entry_down and is_within_band(cy, LINE_Y_END) and obj_id not in logged_down:
                speed = calculate_speed(entry_down[obj_id], frame_counter, fps)
                label = classify_speed(speed)
                draw_vehicle_info(frame, (x1, y1, x2, y2), obj_id, label)
                logged_down.add(obj_id)
                if label == "Overspeeding vehicle":
                    alert_banner = 10

            if is_within_band(cy, LINE_Y_END):
                entry_up[obj_id] = frame_counter
            elif obj_id in entry_up and is_within_band(cy, LINE_Y_START) and obj_id not in logged_up:
                speed = calculate_speed(entry_up[obj_id], frame_counter, fps)
                label = classify_speed(speed)
                draw_vehicle_info(frame, (x1, y1, x2, y2), obj_id, label)
                logged_up.add(obj_id)
                if label == "Overspeeding vehicle":
                    alert_banner = 10

        draw_lines_and_counts(frame, len(logged_down), len(logged_up), alert_banner > 0)
        if alert_banner > 0:
            alert_banner -= 1

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()