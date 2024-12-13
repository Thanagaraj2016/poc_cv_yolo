import cv2
from ultralytics import YOLO  # Ensure YOLOv11 is correctly installed
import numpy as np

# Load YOLOv11 model
model = YOLO("yolo11l.pt")  # Replace with your YOLOv11 model path

# Define parking lot coordinates (replace with your parking lot coordinates)
# Each parking space is a bounding box: [x1, y1, x2, y2]
parking_spaces = [
    [50, 100, 150, 200],
    [200, 100, 300, 200],
    [350, 100, 450, 200],
    # Add more parking spaces as needed
]

# Initialize parking occupancy
parking_status = {i: False for i in range(len(parking_spaces))}  # False = empty, True = occupied

# Define a function to check if a vehicle is in a parking space
def is_vehicle_in_parking_space(vehicle_bbox, parking_space):
    vx1, vy1, vx2, vy2 = vehicle_bbox
    px1, py1, px2, py2 = parking_space
    return vx1 < px2 and vx2 > px1 and vy1 < py2 and vy2 > py1

# Initialize video capture (live video or video file)
video_path = 0  # Use 0 for webcam or replace with a video file path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the frame
    results = model.predict(frame, conf=0.5)  # Adjust confidence threshold as needed
    detections = results[0].boxes.xyxy.cpu().numpy()  # Detected objects (x1, y1, x2, y2)

    # Reset parking status
    parking_status = {i: False for i in range(len(parking_spaces))}

    for det in detections:
        x1, y1, x2, y2 = map(int, det)
        # Check if the detected vehicle is in any parking space
        for idx, space in enumerate(parking_spaces):
            if is_vehicle_in_parking_space([x1, y1, x2, y2], space):
                parking_status[idx] = True

        # Draw detected vehicles on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Vehicle", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw parking spaces and status
    for idx, space in enumerate(parking_spaces):
        x1, y1, x2, y2 = space
        color = (0, 0, 255) if parking_status[idx] else (0, 255, 0)  # Red = occupied, Green = empty
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        status = "Occupied" if parking_status[idx] else "Empty"
        cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Parking Management", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
