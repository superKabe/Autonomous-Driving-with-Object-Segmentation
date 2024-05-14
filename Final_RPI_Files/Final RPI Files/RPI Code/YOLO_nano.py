from ultralytics import YOLO 
import cv2


model = YOLO("yolov8n.pt")

# cap = cv2.VideoCapture(0)  # Use 0 for the primary camera
model.predict(source=2, show=True)

# import cv2
# import numpy as np
# import time

# # Load YOLO Nano model and configuration
# net = cv2.dnn.readNet("yolonano.weights", "yolonano.cfg")
# classes = []
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Set backend and target to use OpenCV's DNN module with OpenVINO
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# # Open video capture object
# cap = cv2.VideoCapture(2)  # Use 0 for the primary camera

# # Get the width and height of the video capture
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Initialize FPS calculation
# start_time = time.time()
# frame_counter = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to blob
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

#     # Set the input to the neural network
#     net.setInput(blob)

#     # Perform a forward pass of the YOLO Nano model
#     start_inference = time.time()
#     outs = net.forward()
#     end_inference = time.time()

#     # Process the detections
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:  # Adjust confidence threshold as needed
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Display FPS
#     frame_counter += 1
#     if frame_counter >= 10:
#         fps = frame_counter / (time.time() - start_time)
#         print("FPS:", fps)
#         frame_counter = 0
#         start_time = time.time()

#     # Display the resulting frame
#     cv2.imshow('Object Detection', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release video capture object and close display window
# cap.release()
# cv2.destroyAllWindows()
