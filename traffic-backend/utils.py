import cv2

def estimate_speed(frame_gap, distance_pixels, mpp, frame_rate):
    return (mpp * distance_pixels) / (frame_gap * 3 / frame_rate) * 3.6

def draw_annotations(frame, bbox, track_id, speed=None, label=None):
    x1, y1, x2, y2 = bbox
    x_mid = (x1 + x2) // 2,
    y_mid = (y1 + y2) // 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

    if speed is not None:
        cv2.putText(frame, f"{int(speed)} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1)

    if label:
        cv2.putText(frame, label, (x2, y2), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 255), 2)

    return x_mid, y_mid

def draw_crossing_lines(frame, y1, y2):
    cv2.line(frame, (274, y1), (814, y1), (255, 255, 255), 1)
    cv2.line(frame, (177, y2), (927, y2), (255, 255, 255), 1)

def draw_direction_count(frame, down_count, up_count):
    cv2.putText(frame, f'Count of vehicles going down: {down_count}', (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Count of vehicles going up: {up_count}', (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)