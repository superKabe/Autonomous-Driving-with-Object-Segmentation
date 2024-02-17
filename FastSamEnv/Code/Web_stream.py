"""
This code captures a video stream from a given URL, decodes the MJPEG stream, and displays the frames in real-time.
It continuously reads data from the stream, identifies JPEG frames, decodes them, and displays them using OpenCV.
"""

import cv2
import urllib.request
import numpy as np

url = 'http://192.168.1.70:8000/stream.mjpg'

stream = urllib.request.urlopen(url)
bytes_array = bytes()

while True:
    bytes_array += stream.read(1024)
    a = bytes_array.find(b'\xff\xd8')
    b = bytes_array.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes_array[a:b+2]
        bytes_array = bytes_array[b+2:]
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Video Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
