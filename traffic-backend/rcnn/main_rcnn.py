import cv2
import torch
from torchvision.transforms import functional as F
from tracker import update_tracker
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

model.eval()

cap = cv2.VideoCapture('veh2.mp4')

cy1, cy2, offset = 322, 368, 6
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame_resized = cv2.resize(frame, (1020, 500))
    tensor_frame = F.to_tensor(frame_resized)

    with torch.no_grad():
        predictions = model([tensor_frame])[0]

    valid_detections = []
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score > 0.6 and label.item() in {3, 6, 8}:
            x1, y1, x2, y2 = map(int, box.tolist())
            valid_detections.append([x1, y1, x2, y2])

    tracked_boxes = update_tracker(valid_detections)

    for x1, y1, x2, y2, obj_id in tracked_boxes:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_resized, str(obj_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("RCNN Tracking", frame_resized)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()