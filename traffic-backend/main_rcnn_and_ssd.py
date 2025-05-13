import cv2
import time
import csv
import torch
import argparse
from torchvision import models
from torchvision.transforms import functional as F
from tracker import update_tracker

def load_model(model_type):
    if model_type == 'rcnn':
        weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    elif model_type == 'ssd':
        model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    else:
        raise ValueError("Invalid model type.")
    model.eval()
    return model

def filter_detections(predictions, model_type):
    detections = []
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score > (0.6 if model_type == 'rcnn' else 0.3) and label.item() in {3, 6, 8}:
            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append([x1, y1, x2, y2])
    return detections

def run_tracking(model_type, input_video):
    model = load_model(model_type)
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    meters_per_pixel = 0.2
    cy1, cy2, offset = 322, 368, 6
    output = cv2.VideoWriter('output_tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (1020, 500))
    count = 0
    vh_down, counter, vh_up, counter1 = {}, [], {}, []
    with open('car_details_rcnn.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Car ID", "Direction", "Speed (Km/h)", "Timestamp"])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            if count < 10 or count % 3 != 0:
                continue

            frame = cv2.resize(frame, (1020, 500))
            tensor_frame = F.to_tensor(frame)

            with torch.no_grad():
                predictions = model([tensor_frame])[0]

            detections = filter_detections(predictions, model_type)
            tracked = update_tracker(detections)

            for x1, y1, x2, y2, obj_id in tracked:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, str(obj_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if writer:
                    if cy1 - offset < cy < cy1 + offset:
                        vh_down[obj_id] = count
                    if obj_id in vh_down and cy2 - offset < cy < cy2 + offset and obj_id not in counter:
                        speed = (meters_per_pixel * abs(cy2 - cy1)) / ((count - vh_down[obj_id]) * 3 / fps) * 3.6
                        writer.writerow([obj_id, "Down", int(speed), time.strftime("%Y-%m-%d %H:%M:%S")])
                        counter.append(obj_id)
                        cv2.putText(frame, f"{int(speed)} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    if cy2 - offset < cy < cy2 + offset:
                        vh_up[obj_id] = count
                    if obj_id in vh_up and cy1 - offset < cy < cy1 + offset and obj_id not in counter1:
                        speed = (meters_per_pixel * abs(cy2 - cy1)) / ((count - vh_up[obj_id]) * 3 / fps) * 3.6
                        writer.writerow([obj_id, "Up", int(speed), time.strftime("%Y-%m-%d %H:%M:%S")])
                        counter1.append(obj_id)
                        cv2.putText(frame, f"{int(speed)} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
            cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
            d = len(counter) 
            u = len(counter1)
            cv2.putText(frame, 'Going down: ' + str(d+1), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, 'Going up: ' + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            output.write(frame)
            cv2.imshow("Vehicle Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["rcnn", "ssd"])
    parser.add_argument("--video", required=True, help="Path to input video")
    args = parser.parse_args()

    run_tracking(args.model, args.video)