from norfair import Detection, Tracker as NorfairTracker
import numpy as np

tracker = NorfairTracker(
    distance_function="euclidean",
    distance_threshold=40
)

def update_tracker(detections):

    norfair_detections = []
    for x1, y1, x2, y2 in detections:
        x_mid, y_mid = ((x1 + x2) / 2), ((y1 + y2) / 2)
        norfair_detections.append(Detection(points=np.array([[x_mid, y_mid]])))

    tracked_objects = tracker.update(detections=norfair_detections)

    output = []
    for obj in tracked_objects:
        x_mid, y_mid = obj.estimate[0]
        obj_id = obj.id

        size_of_box = 40
        x1 = int(x_mid - size_of_box / 2)
        y1 = int(y_mid - size_of_box / 2)
        x2 = int(x_mid + size_of_box / 2)
        y2 = int(y_mid + size_of_box / 2)

        output.append([x1, y1, x2, y2, obj_id])

    return output
