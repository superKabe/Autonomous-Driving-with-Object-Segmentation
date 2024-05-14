import io
import cv2
import logging
import socketserver
from threading import Condition, Thread
from http import server
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)

PAGE = """\
<html>
<head>
<title>Raspberry Pi - Video Stream</title>
</head>
<body>
<h1>Raspberry Pi - Video Stream</h1>
<img src="stream.mjpg" width="640" height="480">
</body>
</html>
"""

class StreamingOutput:
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        logging.debug("Received GET request: %s", self.path)
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# Function to continuously read frames from webcam and write to StreamingOutput buffer
def stream_frames():
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 10)

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame from webcam")
            break
        output.write(cv2.imencode('.jpg', frame)[1].tobytes())

    cap.release()

# Create StreamingOutput object
output = StreamingOutput()

# Start frame streaming thread
frame_thread = Thread(target=stream_frames)
frame_thread.daemon = True
frame_thread.start()

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    logging.info("Server started at %s:%d", *address)
    server.serve_forever()
finally:
    frame_thread.join()
