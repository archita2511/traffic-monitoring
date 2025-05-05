from norfair import Detection, Tracker
import numpy as np

# Create a BYTE-like tracker using Norfair
tracker = Tracker(distance_function="euclidean", distance_threshold=40)

def update_tracker(detections):
    # Convert YOLO boxes to Norfair detections (center points)
    norfair_detections = []
    for box in detections:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        points = np.array([[cx, cy]])
        norfair_detections.append(Detection(points=points))

    # Update tracker with new detections
    tracked_objects = tracker.update(detections=norfair_detections)

    # Convert back to [x1, y1, x2, y2, id]
    output = []
    for obj in tracked_objects:
        cx, cy = obj.estimate[0]
        obj_id = obj.id

        # Reconstruct a box around the tracked center (optional sizing)
        box_size = 40  # adjust as needed
        x1 = int(cx - box_size / 2)
        y1 = int(cy - box_size / 2)
        x2 = int(cx + box_size / 2)
        y2 = int(cy + box_size / 2)

        output.append([x1, y1, x2, y2, obj_id])

    return output
