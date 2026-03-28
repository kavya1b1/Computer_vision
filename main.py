import cv2
import mediapipe as mp
import numpy as np
import os
import time
from ultralytics import YOLO

# -------------------- LOAD MODEL --------------------
model = YOLO("yolov8n.pt")

# -------------------- SETUP --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

# -------------------- FUNCTIONS --------------------
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = euclidean(mouth[0], mouth[1])
    B = euclidean(mouth[2], mouth[3])
    return A / B

# -------------------- LANDMARKS --------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

# -------------------- PARAMETERS --------------------
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 15

frame_counter = 0
fatigue_score = 0
last_alert_time = 0

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------- YOLO (PHONE DETECTION) --------------------
    results_yolo = model(frame, verbose=False)
    phone_detected = False

    for r in results_yolo:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "cell phone":
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # -------------------- FACE PROCESS --------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "ACTIVE"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # -------------------- EYES --------------------
            left_eye = [landmarks[i] for i in LEFT_EYE]
            right_eye = [landmarks[i] for i in RIGHT_EYE]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

            # -------------------- MOUTH --------------------
            mouth = [landmarks[i] for i in MOUTH]
            mar = mouth_aspect_ratio(mouth)

            # -------------------- HEAD POSE --------------------
            nose = landmarks[1]
            left_face = landmarks[234]
            right_face = landmarks[454]

            face_width = euclidean(left_face, right_face)
            head_ratio = euclidean(nose, left_face) / face_width

            # -------------------- LOGIC --------------------
            if ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                frame_counter = 0

            # Priority logic
            if phone_detected:
                status = "PHONE USAGE"
                fatigue_score += 3

            elif frame_counter >= EAR_CONSEC_FRAMES:
                status = "DROWSY"
                fatigue_score += 2

            elif mar > 0.6:
                status = "YAWNING"
                fatigue_score += 1

            elif head_ratio < 0.35 or head_ratio > 0.65:
                status = "DISTRACTED"
                fatigue_score += 2

            fatigue_score = min(fatigue_score, 100)

            # Draw points
            for p in left_eye + right_eye:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            for p in mouth:
                cv2.circle(frame, p, 2, (255, 0, 0), -1)

            # Display metrics
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # -------------------- RISK LEVEL --------------------
    if fatigue_score < 30:
        risk = "LOW"
        color = (0, 255, 0)
    elif fatigue_score < 60:
        risk = "MEDIUM"
        color = (0, 255, 255)
    else:
        risk = "HIGH"
        color = (0, 0, 255)

    # -------------------- ALERT SYSTEM --------------------
    current_time = time.time()
    if risk == "HIGH" and current_time - last_alert_time > 5:
        os.system("say 'Warning High Fatigue Detected'")
        last_alert_time = current_time

    # -------------------- DISPLAY --------------------
    cv2.putText(frame, f"Status: {status}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.putText(frame, f"Fatigue Score: {fatigue_score}", (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"Risk: {risk}", (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("AI Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()