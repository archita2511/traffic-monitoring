import cv2
import time
import csv
import torch
import argparse
from torchvision import models
from torchvision.transforms import functional as F
from tracker import update_tracker

def initialize_detector(model_name):
    if model_name == 'rcnn':
        weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        net = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    elif model_name == 'ssd':
        net = models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    else:
        raise ValueError("Model must be 'rcnn' or 'ssd'")
    net.eval()
    return net

def extract_vehicle_boxes(preds, model_name):
    results = []
    for box, cls, confidence in zip(preds['boxes'], preds['labels'], preds['scores']):
        if confidence > (0.6 if model_name == 'rcnn' else 0.3) and cls.item() in {3, 6, 8}:
            results.append(list(map(int, box.tolist())))
    return results

def estimate_speed(frame_gap, distance_pixels, mpp, frame_rate):
    return (mpp * distance_pixels) / (frame_gap * 3 / frame_rate) * 3.6

def draw_annotations(frame, bbox, track_id, speed=None):
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if speed is not None:
        cv2.putText(frame, f"{int(speed)} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return center_x, center_y

def draw_crossing_lines(frame, y1, y2):
    cv2.line(frame, (274, y1), (814, y1), (255, 255, 255), 1)
    cv2.line(frame, (177, y2), (927, y2), (255, 255, 255), 1)

def draw_direction_count(frame, down_count, up_count):
    cv2.putText(frame, f'Count of vehicles going down: {down_count}', (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Count of vehicles going up: {up_count}', (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

def process_video(model_name, video_path):
    net = initialize_detector(model_name)
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (1020, 500)
    mpp = 0.2
    line_y1, line_y2, tolerance = 322, 368, 6

    writer = cv2.VideoWriter('output_tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    frame_idx = 0
    down_crossed, up_crossed = [], []
    up_pending, down_pending = {}, {}

    with open(f'vehicle_log_{model_name}.csv', 'w', newline='') as log_file:
        logger = csv.writer(log_file)
        logger.writerow(["Vehicle ID", "Direction", "Speed (Km/h)", "Timestamp"])

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_idx += 1
            if frame_idx < 10 or frame_idx % 3 != 0:
                continue

            frame = cv2.resize(frame, frame_size)
            tensor = F.to_tensor(frame)

            with torch.no_grad():
                outputs = net([tensor])[0]

            boxes = extract_vehicle_boxes(outputs, model_name)
            tracked_vehicles = update_tracker(boxes)

            for x1, y1, x2, y2, track_id in tracked_vehicles:
                cx, cy = draw_annotations(frame, (x1, y1, x2, y2), track_id)

                if line_y1 - tolerance < cy < line_y1 + tolerance:
                    down_pending[track_id] = frame_idx
                if (track_id in down_pending and
                        line_y2 - tolerance < cy < line_y2 + tolerance and
                        track_id not in down_crossed):
                    spd = estimate_speed(frame_idx - down_pending[track_id], abs(line_y2 - line_y1), mpp, fps)
                    logger.writerow([track_id, "Down", int(spd), time.strftime("%Y-%m-%d %H:%M:%S")])
                    down_crossed.append(track_id)
                    draw_annotations(frame, (x1, y1, x2, y2), track_id, spd)

                if line_y2 - tolerance < cy < line_y2 + tolerance:
                    up_pending[track_id] = frame_idx
                if (track_id in up_pending and
                        line_y1 - tolerance < cy < line_y1 + tolerance and
                        track_id not in up_crossed):
                    spd = estimate_speed(frame_idx - up_pending[track_id], abs(line_y2 - line_y1), mpp, fps)
                    logger.writerow([track_id, "Up", int(spd), time.strftime("%Y-%m-%d %H:%M:%S")])
                    up_crossed.append(track_id)
                    draw_annotations(frame, (x1, y1, x2, y2), track_id, spd)

            draw_crossing_lines(frame, line_y1, line_y2)
            draw_direction_count(frame, len(down_crossed) + 1, len(up_crossed))

            writer.write(frame)
            cv2.imshow("Tracking Output", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["rcnn", "ssd"])
    parser.add_argument("--video", required=True, help="Path to video file")
    args = parser.parse_args()

    process_video(args.model, args.video)