import cv2
import time
import csv
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from tracker import update_tracker

# Load model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

cap = cv2.VideoCapture("veh2.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
meters_per_pixel = 0.2
cy1, cy2, offset = 322, 368, 6
output = cv2.VideoWriter('output_tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (1020, 500))

vh_down, counter = {}, []
vh_up, counter1 = {}, []
count = 0

with open('car_details_rcnn.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Car ID", "Direction", "Speed (Km/h)", "Timestamp"])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        count += 1
        if count < 10 or count % 3 != 0:
            continue

    # Preprocess frame
        frame_resized = cv2.resize(frame, (1020, 500))
        tensor_frame = F.to_tensor(frame_resized)

        # Perform detection
        with torch.no_grad():
            predictions = model([tensor_frame])[0]

        # Filter detections (car=3, bus=6, truck=8)
        valid_boxes = []
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            if score > 0.6 and label.item() in [3, 6, 8]:
                x1, y1, x2, y2 = map(int, box.tolist())
                valid_boxes.append([x1, y1, x2, y2])

        # Track objects
        tracked_objects = update_tracker(valid_boxes)

        # Process each tracked object
        for x1, y1, x2, y2, obj_id in tracked_objects:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, str(obj_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Check for downward movement
            if cy1 - offset < cy < cy1 + offset:
                vh_down[obj_id] = count
            if obj_id in vh_down and cy2 - offset < cy < cy2 + offset and obj_id not in counter:
                speed = (meters_per_pixel * abs(cy2 - cy1)) / ((count - vh_down[obj_id]) * 3 / fps) * 3.6
                writer.writerow([obj_id, "Down", int(speed), time.strftime("%Y-%m-%d %H:%M:%S")])
                counter.append(obj_id)
                cv2.putText(frame_resized, f"{int(speed)} Km/h", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Check for upward movement
            if cy2 - offset < cy < cy2 + offset:
                vh_up[obj_id] = count
            if obj_id in vh_up and cy1 - offset < cy < cy1 + offset and obj_id not in counter1:
                speed = (meters_per_pixel * abs(cy2 - cy1)) / ((count - vh_up[obj_id]) * 3 / fps) * 3.6
                writer.writerow([obj_id, "Up", int(speed), time.strftime("%Y-%m-%d %H:%M:%S")])
                counter1.append(obj_id)
                cv2.putText(frame_resized, f"{int(speed)} Km/h", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw guide lines
        cv2.line(frame_resized, (274, cy1), (814, cy1), (255, 255, 255), 1)
        cv2.line(frame_resized, (177, cy2), (927, cy2), (255, 255, 255), 1)

        output.write(frame_resized)
        cv2.imshow("RCNN Tracking", frame_resized)
        if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
output.release()
cv2.destroyAllWindows()
