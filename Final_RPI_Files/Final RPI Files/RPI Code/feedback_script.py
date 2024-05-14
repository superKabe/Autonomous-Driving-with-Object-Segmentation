import cv2
import numpy as np
import urllib.request

# URL of the video stream
url = 'http://192.168.1.13:5555/video_feed'  # Home WiFi
# url = 'http://192.168.137.99:5555/video_feed'  # Hotspot


# Function to capture and display the video stream
def capture_and_display_stream(url):
    stream = urllib.request.urlopen(url)  # Open the video stream

    while True:
        try:
            bytes_array = bytes()  # Initialize byte array for frame data
            buffer_size = 8198  # Adjust buffer size as needed

            # Read frame data from the stream
            while True:
                bytes_array += stream.read(buffer_size)
                a = bytes_array.find(b'\xff\xd8')  # Find start of frame
                b = bytes_array.find(b'\xff\xd9')  # Find end of frame
                if a != -1 and b != -1:
                    jpg = bytes_array[a:b+2]  # Extract frame data
                    bytes_array = bytes_array[b+2:]  # Update byte array

                    if len(jpg) > 0:
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)  # Decode frame
                        break

            # Display the frame
            cv2.imshow('Video Stream', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print("Error:", e)
            cv2.destroyAllWindows()
            break

        # Small delay to allow the window to update
        cv2.waitKey(10)

    # Release resources
    cv2.destroyAllWindows()

# Run the function to capture and display the video stream
capture_and_display_stream(url)
