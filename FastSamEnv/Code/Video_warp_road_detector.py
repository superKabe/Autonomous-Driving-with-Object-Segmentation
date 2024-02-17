"""
This code processes a video stream by detecting lines using Hough transform after preprocessing each frame.
Preprocessing involves converting frames to grayscale, equalizing histograms, and applying adaptive thresholding.
It also allows the user to adjust perspective transformation parameters via trackbars in real-time for better line detection.
"""

import cv2
import numpy as np

def initialize_trackbars(initial_trackbar_vals, wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    for name, val in zip(["Width Top", "Height Top", "Width Bottom", "Height Bottom"], initial_trackbar_vals):
        cv2.createTrackbar(name, "Trackbars", val, wT // 2 if "Width" in name else hT, nothing)

def val_trackbars(wT=480, hT=240):
    width_top = cv2.getTrackbarPos("Width Top", "Trackbars")
    height_top = cv2.getTrackbarPos("Height Top", "Trackbars")
    width_bottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    height_bottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    return np.float32([(width_top, height_top), (wT-width_top, height_top), (width_bottom, height_bottom), (wT-width_bottom, height_bottom)])

def warp_img(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts2, pts1) if inv else cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (w, h))

def canny_edge(video_frame):
    canny_low_threshold, canny_high_threshold, kernel_size = 70, 80, 29
    gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)
    # blurred_frame = cv2.GaussianBlur(equalized_frame, (kernel_size, kernel_size), 0)
    
    # Apply simple thresholding to convert the frame to black and white
    _, binary_frame = cv2.threshold(equalized_frame, 155, 255, cv2.THRESH_BINARY)

    return cv2.Canny(binary_frame, canny_low_threshold, canny_high_threshold)

def filter_vertical_lines(lines, angle_threshold):
    if lines is not None:  # Add a check to handle NoneType
        return [line for line in lines if abs(np.degrees(np.arctan((line[0][3] - line[0][1]) / (line[0][2] - line[0][0] + 1e-5)))) < angle_threshold]
    else:
        return []

def plot_hough_lines(image, rho, theta, threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    line_image = np.zeros_like(image)
    if lines is not None:
        vertical_lines = filter_vertical_lines(lines, angle_threshold=360)
        for line in vertical_lines:
            cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 0, 0), 2)
    return line_image

def nothing(a):
    pass

if __name__ == '__main__':
    cap = cv2.VideoCapture('hallway_highway.mp4')
    initial_trackbar_vals = [240, 240, 240, 480]
    initialize_trackbars(initial_trackbar_vals)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to load the frame.")
            break

        points = val_trackbars(wT=frame.shape[1], hT=frame.shape[0])
        hT, wT, _ = frame.shape
        imgWarp = warp_img(frame, points, wT, hT)
        canny_frame = canny_edge(imgWarp)
        hough_lines_image = plot_hough_lines(canny_frame, rho=1, theta=np.pi / 180, threshold=50, min_line_length=200, max_line_gap=90)
        
        # Check if vertical_lines is not empty before processing
        vertical_lines = cv2.HoughLinesP(canny_frame, 1, np.pi / 180, 30, np.array([]), 70, 50)
        if vertical_lines is not None and len(vertical_lines) > 0:
            vertical_lines_filtered = filter_vertical_lines(vertical_lines, angle_threshold=360)
            if vertical_lines_filtered:
                center_of_road = np.mean([(line[0][0] + line[0][2]) / 2 for line in vertical_lines_filtered])
                cv2.line(hough_lines_image, (int(center_of_road), 0), (int(center_of_road), hough_lines_image.shape[0]), (255, 0, 0), 5)

        hough_lines_image_colored = cv2.cvtColor(hough_lines_image, cv2.COLOR_GRAY2BGR)
        combined_image = cv2.addWeighted(imgWarp, 0.8, hough_lines_image_colored, 1, 0)

        cv2.imshow('Video with Hough Lines', combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
