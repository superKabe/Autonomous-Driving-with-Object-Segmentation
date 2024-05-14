"""
This script captures video from a camera, runs a MobileNetV2 model to perform image classification on each frame, and displays the top 3 predictions along with FPS on the video feed.
"""
import time
import torch
import numpy as np
from torchvision import models, transforms
import cv2
import json

# Load the pre-trained MobileNetV2 model
net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)

# Load the ImageNet class index
class_idx_file = "imagenet_class_index.json"
with open(class_idx_file, "r") as f:
    class_idx = json.load(f)

# Preprocess function
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Video capture
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FPS, 36)

# Initialize frame count and time
frame_count = 0
last_logged = time.time()

# Inference loop
with torch.no_grad():
    while True:
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(image_rgb)
        input_batch = input_tensor.unsqueeze(0)
        output = net(input_batch)
        
        top = list(enumerate(output[0].softmax(dim=0)))
        top.sort(key=lambda x: x[1], reverse=True)
        predictions = [f"{class_idx[str(idx)][1]}: {val.item()*100:.2f}%" for idx, val in top[:3]]
        print(predictions)

        # Calculate and write FPS onto the frame
        frame_count += 1
        now = time.time()
        fps = frame_count / (now - last_logged)
        print(fps)

        # Display the frame (convert RGB to BGR before displaying)
        cv2.imshow("Video", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        
        # Exit on pressing 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
