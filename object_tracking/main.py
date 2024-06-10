import cv2
import numpy as np
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=70, varThreshold=400)

# Diccionario para registrar los IDs de los contornos detectados
detected_contours = {}

def merge_contours(contours, threshold=50, distance_threshold=100):
    merged_contours = []
    used = [False] * len(contours)

    for i, cnt1 in enumerate(contours):
        if used[i]:
            continue
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        rect1 = np.array([x1, y1, x1 + w1, y1 + h1])

        for j, cnt2 in enumerate(contours):
            if i == j or used[j]:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
            rect2 = np.array([x2, y2, x2 + w2, y2 + h2])

            # Check distance between the bounding boxes
            if np.linalg.norm(rect1[:3] - rect2[:3]) < threshold:
                # Calculate centroids of the contours
                centroid1 = (x1 + w1 // 2, y1 + h1 // 2)
                centroid2 = (x2 + w2 // 2, y2 + h2 // 2)

                # Calculate distance between centroids
                centroid_distance = np.linalg.norm(np.array(centroid1) - np.array(centroid2))

                # If centroids are close but distance between contours is within threshold, merge
                if centroid_distance < distance_threshold:
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1 + w1, x2 + w2) - new_x
                    new_h = max(y1 + h1, y2 + h2) - new_y
                    merged_rect = np.array([new_x, new_y, new_x + new_w, new_y + new_h])
                    rect1 = merged_rect
                    used[j] = True

        used[i] = True
        merged_contours.append(rect1)

    return merged_contours

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape

    # Resize the frame to a higher resolution to capture more details
    frame = cv2.resize(frame, (1280, 720))

    # Adjust the Region of Interest (ROI)
    roi = frame[10: 1000, 560: 800]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]

    # Merge closely spaced contours
    merged_contours = merge_contours(filtered_contours)

    detections = []
    for rect in merged_contours:
        x, y, x2, y2 = rect
        w, h = x2 - x, y2 - y
        detections.append([x, y, w, h])

    # 2. Object Tracking
    # Registrar el tiempo de inicio antes de actualizar el rastreador
    start_time = cv2.getTickCount()
    boxes_ids = tracker.update(detections)
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000

    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Verificar si este ID de contorno ya se ha detectado
        if id not in detected_contours:
            print("Objeto {} detectado en {:.2f} ms".format(id, elapsed_time))
            detected_contours[id] = True

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
