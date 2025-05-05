from norfair import Detection, Tracker
from norfair.distances import iou
import numpy as np

tracker = Tracker(distance_function=iou, distance_threshold=0.3)

def update_tracker(detections):
    norfair_detections = []
    for x1, y1, x2, y2 in detections:
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

        # Dummy point just to satisfy Norfair's internal checks
        dummy_points = np.array([[x1, y1]])  # Single point

        norfair_detections.append(
            Detection(
                points=dummy_points,       # 👈 Required (even if unused)
                scores=np.array([1.0]),    # 👈 Also required
                data=bbox                  # 👈 Used by IoU
            )
        )

    tracked_objects = tracker.update(detections=norfair_detections)

    output = []
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.last_detection.data
        output.append([int(x1), int(y1), int(x2), int(y2), obj.id])

    return output
