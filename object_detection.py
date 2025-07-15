import torch
import cv2
import numpy as np
import os

# i Load the YOLOv5 model from the pytroch
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=False)

# this is like a folder to save detected frames
os.makedirs('detected_frames', exist_ok=True)

# its to make and save the videos
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

# Use webcam
cap = cv2.VideoCapture(0)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # it is fit the model to the frame
    results = model(frame)
    

    # Print detected objects’ names and confidence scores
    for *box, conf, cls in results.xyxy[0]:
        label = results.names[int(cls)]
        print(f"Detected: {label:<15} | Confidence: {conf:.2f}")
        
    # Render results on the frame
    annotated_frame = results.render()[0]  # results.render() returns a list of images

    # Save frames that contain detections with confidence > 0.50
    if len(results.xyxy[0]) > 0 and float(results.xyxy[0][0][4]) > 0.50:
        cv2.imwrite(f'detected_frames/frame_{frame_count:05d}.jpg', annotated_frame)

    # Initialize VideoWriter after we know frame size
    if out is None:
        height, width = annotated_frame.shape[:2]
        out = cv2.VideoWriter('output_with_detections.mp4', fourcc, 20.0, (width, height))

    # Record annotated video
    out.write(annotated_frame)

    cv2.imshow('YOLOv5 Detection', annotated_frame)

    # Break with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()