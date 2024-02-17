""" 
This script captures a video stream from a Raspberry Pi camera, applies perspective transformation based 
on trackbar values, performs Canny edge detection and Hough line detection, and calculates the center of 
the road to assist in steering adjustment for autonomous vehicles.
"""

import cv2
import numpy as np
import picamera
import picamera.array
import time

def nothing(a):
    pass

def initializeTrackbars(intialTracbarVals,wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)

def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points

def warpImg (img,points,w,h,inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp

def canny_edge(video_frame):
    canny_low_threshold = 20
    canny_high_threshold = 40
    kernel_size = 19

    # Applying Histogram Equalization for contrast enhancement
    gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)

    # Applying Gaussian Blur
    blurred_frame = cv2.GaussianBlur(equalized_frame, (kernel_size, kernel_size), 0)

    # Canny Edge Detection on the blurred and equalized frame
    canny_frame = cv2.Canny(blurred_frame, canny_low_threshold, canny_high_threshold)
    return canny_frame

def filter_vertical_lines(lines, angle_threshold):
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate the slope of the line
        slope = (y2 - y1) / (x2 - x1 + 1e-5)  # Adding a small value to avoid division by zero

        # Check if the line is approximately vertical
        if abs(np.degrees(np.arctan(slope))) < angle_threshold:
            vertical_lines.append(line)

    return vertical_lines

def plot_hough_lines(image, rho, theta, threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    line_image = np.zeros_like(image)

    if lines is not None:
        # Filter primarily vertical lines
        vertical_lines = filter_vertical_lines(lines, angle_threshold=360)

        for line in vertical_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return line_image

if __name__ == '__main__':
    with picamera.PiCamera() as camera:
        # Set camera resolution (adjust as needed)
        camera.resolution = (640, 480)
        # Allow camera to warm up
        time.sleep(2)

        intialTracbarVals = [240,240,240,480]
        initializeTrackbars(intialTracbarVals)

        while True:
            with picamera.array.PiRGBArray(camera) as stream:
                # Capture a frame from the camera
                camera.capture(stream, format='bgr')
                frame = stream.array

            if frame is None:
                print("Error: Unable to load the frame.")
                break

            points = valTrackbars(wT=frame.shape[1], hT=frame.shape[0])

            hT, wT, c = frame.shape
            imgWarp = warpImg(frame, points, wT, hT)

            canny_frame = canny_edge(imgWarp)

            # Create a blank canvas with the same dimensions as the original frame
            hough_lines_image = np.zeros_like(imgWarp)

            # Plot Hough lines on the blank canvas
            hough_lines_image = plot_hough_lines(canny_frame, rho=1, theta=np.pi / 180, threshold=30, min_line_length=70, max_line_gap=50)

            # Calculate the center of the road
            vertical_lines = filter_vertical_lines(cv2.HoughLinesP(canny_frame, 1, np.pi / 180, 30, np.array([]), 70, 50), angle_threshold=360)
            if vertical_lines:
                # Calculate the average x-coordinate of the vertical lines
                center_of_road = np.mean([((line[0][0] + line[0][2]) / 2) for line in vertical_lines])

                # Draw a line at the center of the road
                cv2.line(hough_lines_image, (int(center_of_road), 0), (int(center_of_road), hough_lines_image.shape[0]), (255, 0, 0), 5)

                # Adjust the car's steering based on the center of the road
                # Implement steering adjustment logic here

            # Convert the Hough lines image to three channels
            hough_lines_image_colored = cv2.cvtColor(hough_lines_image, cv2.COLOR_GRAY2BGR)

            # Combine the original frame with the Hough lines image
            combined_image = cv2.addWeighted(imgWarp, 0.8, hough_lines_image_colored, 1, 0)

            # Display the combined image
            cv2.imshow('Video with Hough Lines', combined_image)

            # Stop the video if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Clear the stream in preparation for the next frame
            stream.truncate(0)

        cv2.destroyAllWindows()
