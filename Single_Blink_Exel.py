import face_recognition
import cv2
import numpy as np
import pickle
import pandas as pd
import dlib
from datetime import datetime
from scipy.spatial import distance
import os

# Load known faces
with open("face_encodings.pkl", "rb") as file:
    known_face_encodings, known_face_names = pickle.load(file)

# Initialize attendance file
attendance_file = "attendance.csv"

# Load existing attendance file or create a new one
if os.path.exists(attendance_file):
    df = pd.read_csv(attendance_file)
else:
    df = pd.DataFrame(columns=["Name", "Time"])

# Initialize Dlib's face detector and facial landmark predictor
model_path = "D:/Resources For Projects/Classroom_Attendance_System/shape_predictor_68_face_landmarks.dat"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Error: {model_path} not found! Please check the file path.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Blink detection parameters
EYE_AR_THRESHOLD = 0.2  # Threshold to determine if eyes are closed
BLINK_FRAMES = 3  # Number of consecutive frames required for a blink
blink_counter = {}
blink_detected = {}

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    detected_faces = detector(gray)

    # Loop through detected faces
    for face_encoding, face_location, face in zip(face_encodings, face_locations, detected_faces):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if matches:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Initialize blink counter for this person if not already set
        if name not in blink_counter:
            blink_counter[name] = 0
            blink_detected[name] = False

        # Get facial landmarks
        landmarks = predictor(gray, face)

        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0  # Average of both eyes

        # Blink detection logic
        if ear < EYE_AR_THRESHOLD:
            blink_counter[name] += 1
        else:
            if blink_counter[name] >= BLINK_FRAMES:
                blink_detected[name] = True
                print(f"✅ Blink detected for {name}")

            blink_counter[name] = 0  # Reset counter when eyes open

        # Mark attendance only if a blink is detected
        if blink_detected[name]:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if name not in df["Name"].values:
                new_entry = pd.DataFrame([[name, current_time]], columns=["Name", "Time"])
                df = pd.concat([df, new_entry], ignore_index=True)
                df.to_csv(attendance_file, index=False)
                print(f"✅ Attendance marked for {name} at {current_time}")

            blink_detected[name] = False  # Reset for next detection

        # Draw a rectangle and label on the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Attendance System", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
print("Attendance system closed.")
