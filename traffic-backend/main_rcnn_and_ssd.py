import cv2
import time
import csv
import torch
import argparse
from torchvision import models
from torchvision.transforms import functional as F
from tracker import update_tracker
from utils import estimate_speed, draw_annotations, draw_crossing_lines, draw_direction_count

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

def process_video(model_name, video_path):
    net = initialize_detector(model_name)
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (1020, 500)
    mpp = 0.2
    line_y1 = 322
    line_y2 = 368
    tolerance = 6

    writer = cv2.VideoWriter('output_tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    frame_idx = 0
    down_crossed = []
    up_crossed = []
    up_pending = {}
    down_pending = {}

    with open(f'vehicle_log_{model_name}.csv', 'w', newline='') as log_file:
        logger = csv.writer(log_file)
        logger.writerow(["Vehicle ID", "Vehicle Direction", "Speed in Km/hr", "Timestamp"])

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_idx += 1
            if frame_idx < 10 or frame_idx % 3 != 0:
                continue

            resized_frame = cv2.resize(frame, frame_size)
            tensor = F.to_tensor(resized_frame)

            with torch.no_grad():
                outputs = net([tensor])[0]

            boxes = extract_vehicle_boxes(outputs, model_name)
            tracked_vehicles = update_tracker(boxes)

            for x1, y1, x2, y2, track_id in tracked_vehicles:
                cx, cy = draw_annotations(resized_frame, (x1, y1, x2, y2), track_id)

                if line_y1 - tolerance < cy < line_y1 + tolerance:
                    down_pending[track_id] = frame_idx
                if (track_id in down_pending and
                        line_y2 - tolerance < cy < line_y2 + tolerance and
                        track_id not in down_crossed):
                    spd = estimate_speed(frame_idx - down_pending[track_id], abs(line_y2 - line_y1), mpp, fps)
                    logger.writerow([track_id, "Down", int(spd), time.strftime("%Y-%m-%d %H:%M:%S")])
                    down_crossed.append(track_id)
                    draw_annotations(resized_frame, (x1, y1, x2, y2), track_id, spd)

                if line_y2 - tolerance < cy < line_y2 + tolerance:
                    up_pending[track_id] = frame_idx
                if (track_id in up_pending and
                        line_y1 - tolerance < cy < line_y1 + tolerance and
                        track_id not in up_crossed):
                    spd = estimate_speed(frame_idx - up_pending[track_id], abs(line_y2 - line_y1), mpp, fps)
                    logger.writerow([track_id, "Up", int(spd), time.strftime("%Y-%m-%d %H:%M:%S")])
                    up_crossed.append(track_id)
                    draw_annotations(resized_frame, (x1, y1, x2, y2), track_id, spd)

            draw_crossing_lines(resized_frame, line_y1, line_y2)
            draw_direction_count(resized_frame, len(down_crossed) + 1, len(up_crossed))

            writer.write(resized_frame)
            cv2.imshow("Tracking Output", resized_frame)
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