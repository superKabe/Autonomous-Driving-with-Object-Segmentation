""" 
This code processes a video stream by detecting lines using Hough transform after preprocessing each frame.
Preprocessing steps include converting the frame to grayscale, performing histogram equalization,
applying Gaussian blur, and adaptive thresholding to create a binary image. The detected lines are then
overlaid onto the original frame, and the resulting image is displayed in real-time. 
"""


import cv2
import numpy as np

def preprocess_image(img):
    # Grayscale conversion, histogram equalization, and Gaussian blur
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)
    blurred_frame = cv2.GaussianBlur(equalized_frame, (5, 5), 0)

    # Adaptive thresholding
    binary_image = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return binary_image

def plot_hough_lines(image, rho, theta, threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return line_image

if __name__ == '__main__':
    cap = cv2.VideoCapture('hallway_highway.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to load the frame.")
            break

        # Preprocess the frame: Convert to binary with adaptive thresholding
        binary_image = preprocess_image(frame)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(binary_image, 50, 150)

        # Apply Hough transform to detect lines
        hough_lines_image = plot_hough_lines(edges, rho=1, theta=np.pi / 180, threshold=100, min_line_length=150, max_line_gap=20)

        # Combine the original frame with the Hough lines image
        combined_image = cv2.addWeighted(frame, 0.8, cv2.cvtColor(hough_lines_image, cv2.COLOR_GRAY2BGR), 1, 0)

        cv2.imshow('Video with Hough Lines', combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
