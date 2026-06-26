from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import time

app = Flask(__name__)

picam2 = Picamera2()

picam2.configure(
    picam2.create_video_configuration(
        main={
            #"size": (1280, 720),
            "size": (1640, 1640),
            #"size": (640, 640),
            "format": "RGB888"
        }
    )
)

print(picam2.camera_configuration())

picam2.start()

time.sleep(2)

def generate():
    while True:
        frame = picam2.capture_array()

        _, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )

@app.route("/stream")
def stream():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        threaded=True
    )