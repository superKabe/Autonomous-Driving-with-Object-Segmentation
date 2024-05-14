import cv2
import numpy as np
from picamera2 import Picamera2
from Motor import *
import datetime
import threading

# Create a motor object
PWM = Motor()

# Create a threading lock
motor_lock = threading.Lock()

# Global variable for the low threshold value
low_threshold = 165

# Flag to enable/disable check_vehicle_position
enable_check_vehicle_position = False

def canny_edge(video_frame):
    global low_threshold  # Declare the global variable

    gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)  # Reduce blur kernel size for sharper edges
    ret, thresh_frame = cv2.threshold(blurred_frame, low_threshold, 255, cv2.THRESH_BINARY_INV)  # Invert threshold to detect black lines on white background
    canny_frame = cv2.Canny(thresh_frame, 50, 150)  # Adjust thresholds for Canny edge detection
    return canny_frame

def filter_vertical_lines(lines, angle_threshold):
    if lines is not None:
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if np.abs(angle - 90) < angle_threshold:
                vertical_lines.append(line)
        return vertical_lines
    else:
        return []

# Function to move left
def move_left():
    def left_movement():
        PWM.setMotorModel(-1500, -1500, 1500, 1500)  # Left Upper, Left Lower, Right Upper, Right Lower
        # Pause for n seconds
        time.sleep(1)
        PWM.setMotorModel(0, 0, 0, 0)

    # Create a new thread for left movement
    left_thread = threading.Thread(target=left_movement)
    # Start the thread
    left_thread.start()

# Function to move forward
def move_fwd():
    def fwd_movement():
        PWM.setMotorModel(1500, 1500, 1500, 1500)  # Left Upper, Left Lower, Right Upper, Right Lower
        # Pause for n seconds
        time.sleep(1)
        PWM.setMotorModel(0, 0, 0, 0)

    # Create a new thread for forward movement
    fwd_thread = threading.Thread(target=fwd_movement)
    # Start the thread
    fwd_thread.start()

# Function to move right
def move_right():
    def right_movement():
        PWM.setMotorModel(1500, 1500, -1500, -1500)  # Left Upper, Left Lower, Right Upper, Right Lower
        # Pause for n seconds
        time.sleep(1)
        PWM.setMotorModel(0, 0, 0, 0)

    # Create a new thread for right movement
    right_thread = threading.Thread(target=right_movement)
    # Start the thread
    right_thread.start()

# Adjust center margin as needed
def check_vehicle_position(line, image_width, center_margin=60):
    # Calculate the x-coordinate of the line's midpoint
    line_midpoint_x = line

    # Calculate the center of the image
    image_center = image_width // 2

    # Check if the line's midpoint is within the center margin of the image center
    if abs(line_midpoint_x - image_center) <= center_margin:
        # Car is centered
        move_fwd()
    else:
        if line_midpoint_x - image_center > 0:
            # Car is Left of Center
            move_left()  # Currently set to inverse of original

        elif line_midpoint_x - image_center < 0:
            # Car is Right of Center
            move_right()  # Currently set to inverse of original

# Function to update low threshold value
def update_low_threshold(value):
    global low_threshold
    low_threshold = value

if __name__ == '__main__':
    picam2 = Picamera2()
    width = 640
    height = 480
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (width, height)}))
    picam2.start()

    # Variables for tracking the center position over multiple frames
    center_history = []
    history_length = 10

    # Create a window to display the video
    cv2.namedWindow('Canny Edge Detection with Hough Lines')

    # Create trackbar to adjust low threshold value
    cv2.createTrackbar('Low Threshold', 'Canny Edge Detection with Hough Lines', low_threshold, 255, update_low_threshold)

    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 0)
        roi = frame

        # Perform Canny edge detection on ROI
        canny_frame = canny_edge(roi)

        # Perform Hough line detection only on the ROI
        try:
            hough_lines = cv2.HoughLinesP(canny_frame, 1, np.pi / 180, 30, minLineLength=100, maxLineGap=40)
        except:
            hough_lines = None

        if hough_lines is not None:
            vertical_lines_filtered = filter_vertical_lines(hough_lines, angle_threshold=60)

            if vertical_lines_filtered:
                vertical_lines_filtered.sort(key=lambda x: np.linalg.norm(np.array(x[0][:2]) - np.array(x[0][2:])), reverse=True)
                top_lines = vertical_lines_filtered[:4]

                for line in top_lines:
                    cv2.line(canny_frame, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 0, 0), 2)

                # Calculate the average center position
                center_of_road = np.mean([(line[0][0] + line[0][2]) / 2 for line in top_lines])

                # Append the current center position to the history
                center_history.append(center_of_road)

                # Keep only the last `history_length` values in the history
                if len(center_history) > history_length:
                    center_history = center_history[-history_length:]

                # Calculate the average center position over the history
                avg_center_of_road = int(np.mean(center_history))

                # Draw the center line based on the average position
                #cv2.line(canny_frame, (avg_center_of_road, 0), (avg_center_of_road, canny_frame.shape[0]), (255, 0, 0), 5)

                if enable_check_vehicle_position:
                    check_vehicle_position(avg_center_of_road, width)



        # Check for 'D' key press to toggle the state of driving
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d') or key == ord('D'):
            print("Drive mode changed")
            enable_check_vehicle_position = not enable_check_vehicle_position

        # Display the state of check_vehicle_position
        state_text = "Vehicle Movement: {}".format("Enabled" if enable_check_vehicle_position else "Disabled")
        cv2.putText(canny_frame, state_text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Display the Canny edge detection output
        cv2.imshow('Canny Edge Detection with Hough Lines', canny_frame)

        if key == ord('q') or key == ord('Q'):
            print("Quitting...")
            break

    picam2.stop()
    cv2.destroyAllWindows()
