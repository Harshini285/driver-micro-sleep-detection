import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import math
import time
import threading
import winsound
from collections import deque

# ===================== CONSTANTS =====================
MAR_THRESHOLD = 0.6
MICROSLEEP_TIME = 2.0
BLINK_WINDOW = 60
BLINK_RATE_THRESHOLD = 20
STATE_HOLD_TIME = 1.0
CALIBRATION_TIME = 5.0
FACE_LOST_TIME = 2.0
ALARM_COOLDOWN = 1.0   # seconds

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [61, 81, 13, 291, 311, 402]

# ===================== CLASS =====================
class DriverMonitor:
    def __init__(self):
        self.blink_timestamps = deque()
        self.eyes_closed_prev = False
        self.eye_close_start = None

        self.face_lost_start = None

        self.state = "CALIBRATING"
        self.state_change_time = time.time()

        self.ear_threshold = None
        self.calibration_values = []
        self.calibration_start = time.time()

        self.last_alarm_time = 0.0

    # ---------- Math ----------
    def euclidean_dist(self, p1, p2):
        return math.dist(p1, p2)

    def eye_aspect_ratio(self, eye):
        A = self.euclidean_dist(eye[1], eye[5])
        B = self.euclidean_dist(eye[2], eye[4])
        C = self.euclidean_dist(eye[0], eye[3])
        return 0 if C == 0 else (A + B) / (2.0 * C)

    def mouth_aspect_ratio(self, mouth):
        A = self.euclidean_dist(mouth[1], mouth[5])
        B = self.euclidean_dist(mouth[2], mouth[4])
        C = self.euclidean_dist(mouth[0], mouth[3])
        return 0 if C == 0 else (A + B) / (2.0 * C)

    # ---------- Calibration ----------
    def calibrate(self, ear):
        self.calibration_values.append(ear)
        if time.time() - self.calibration_start >= CALIBRATION_TIME:
            avg_ear = sum(self.calibration_values) / len(self.calibration_values)
            self.ear_threshold = 0.7 * avg_ear
            self.state = "NORMAL"
            self.state_change_time = time.time()   

    # ---------- Alarm ----------
    def trigger_alarm(self):
        now = time.time()
        if now - self.last_alarm_time > ALARM_COOLDOWN:
            self.last_alarm_time = now
            threading.Thread(
                target=lambda: winsound.Beep(1200, 700),
                daemon=True
            ).start()

    # ---------- Face Lost ----------
    def handle_face_lost(self):
        now = time.time()
        if self.face_lost_start is None:
            self.face_lost_start = now
        elif now - self.face_lost_start >= FACE_LOST_TIME:
            self.state = "DANGER"
            self.trigger_alarm()

    # ---------- State Update ----------
    def update_state(self, ear, mar):
        now = time.time()

        # ---- Blink & Eye Closure ----
        if ear < self.ear_threshold:
            if not self.eyes_closed_prev:
                self.blink_timestamps.append(now)
                self.eyes_closed_prev = True
                self.eye_close_start = now
        else:
            self.eyes_closed_prev = False
            self.eye_close_start = None

        # ---- Blink Rate ----
        while self.blink_timestamps and now - self.blink_timestamps[0] > BLINK_WINDOW:
            self.blink_timestamps.popleft()

        blink_rate = len(self.blink_timestamps)

        # ---- Emergency: Micro-sleep (IMMEDIATE) ----
        if self.eye_close_start and (now - self.eye_close_start) >= MICROSLEEP_TIME:
            self.state = "DANGER"
            self.trigger_alarm()
            return blink_rate

        # ---- Non-emergency states ----
        new_state = "DROWSY" if (blink_rate > BLINK_RATE_THRESHOLD or mar > MAR_THRESHOLD) else "NORMAL"

        if new_state != self.state:
            if now - self.state_change_time > STATE_HOLD_TIME:
                self.state = new_state
                self.state_change_time = now
        else:
            self.state_change_time = now

        return blink_rate

# ===================== MEDIAPIPE =====================
base_options = python.BaseOptions(model_asset_path="models/face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

monitor = DriverMonitor()
cap = cv2.VideoCapture(0)

# ===================== MAIN LOOP =====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if not result.face_landmarks:
        monitor.handle_face_lost()
        cv2.putText(frame, "LOOK AWAY DETECTED", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        monitor.face_lost_start = None
        face = result.face_landmarks[0]

        left_eye = [(int(face[i].x * w), int(face[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(face[i].x * w), int(face[i].y * h)) for i in RIGHT_EYE]
        mouth = [(int(face[i].x * w), int(face[i].y * h)) for i in MOUTH]

        ear = (monitor.eye_aspect_ratio(left_eye) +
               monitor.eye_aspect_ratio(right_eye)) / 2.0
        mar = monitor.mouth_aspect_ratio(mouth)

        if monitor.state == "CALIBRATING":
            remaining = int(CALIBRATION_TIME - (time.time() - monitor.calibration_start))
            cv2.putText(frame, f"CALIBRATING... {max(0, remaining)}s",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            monitor.calibrate(ear)
        else:
            blink_rate = monitor.update_state(ear, mar)

            color = (0,255,0) if monitor.state=="NORMAL" else \
                    (0,255,255) if monitor.state=="DROWSY" else (0,0,255)

            cv2.putText(frame, f"STATE: {monitor.state}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Blinks/min: {blink_rate}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.imshow("Driver Micro-Sleep Detection (Final)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
