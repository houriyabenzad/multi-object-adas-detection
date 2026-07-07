
from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

ADAS_CLASSES = [
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "traffic light"
]

video = cv2.VideoCapture("data/road_video.mp4")

if not video.isOpened():
    print("Video not found.")
    exit()

prev_time = time.time()

while True:
    ret, frame = video.read()

    if not ret:
        break

    counts = {
        "car": 0,
        "person": 0,
        "truck": 0,
        "bus": 0,
        "motorcycle": 0,
        "bicycle": 0,
        "traffic light": 0
    }

    results = model.track(frame, persist=True, conf=0.50)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]

            if class_name not in ADAS_CLASSES:
                continue

            counts[class_name] += 1

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            object_id = "N/A"
            if box.id is not None:
                object_id = int(box.id[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{class_name} ID:{object_id} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.rectangle(frame, (10, 10), (360, 245), (40, 40, 40), -1)

    cv2.putText(frame, "ADAS AI TRACKING", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"Cars: {counts['car']}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Persons: {counts['person']}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Trucks: {counts['truck']}", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Buses: {counts['bus']}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Motorcycles: {counts['motorcycle']}", (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Bicycles: {counts['bicycle']}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Traffic lights: {counts['traffic light']}", (20, 225),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (250, 225),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("ADAS AI Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
