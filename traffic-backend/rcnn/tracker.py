from norfair import Detection, Tracker as NorfairTracker
import numpy as np

# Instantiate a Norfair tracker that mimics ByteTrack behavior
tracker = NorfairTracker(
    distance_function="euclidean",
    distance_threshold=40
)

def update_tracker(detections):
    """
    detections: list of [x1, y1, x2, y2] from YOLO
    returns: list of [x1, y1, x2, y2, object_id]
    """
    norfair_detections = []
    for x1, y1, x2, y2 in detections:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        norfair_detections.append(Detection(points=np.array([[cx, cy]])))

    tracked_objects = tracker.update(detections=norfair_detections)

    output = []
    for obj in tracked_objects:
        cx, cy = obj.estimate[0]
        obj_id = obj.id

        # Reconstruct a bounding box around the center
        box_size = 40
        x1 = int(cx - box_size / 2)
        y1 = int(cy - box_size / 2)
        x2 = int(cx + box_size / 2)
        y2 = int(cy + box_size / 2)

        output.append([x1, y1, x2, y2, obj_id])

    return output