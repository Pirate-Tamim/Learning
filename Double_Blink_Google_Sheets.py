import face_recognition
import cv2
import numpy as np
import pickle
import pandas as pd
import dlib
from datetime import datetime
from scipy.spatial import distance
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time

# Load known faces
with open("face_encodings.pkl", "rb") as file:
    known_face_encodings, known_face_names = pickle.load(file)

# Google Sheets API Authentication
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)  # Use your own credentials file
client = gspread.authorize(creds)

# Open the Google Sheet by its name
attendance_sheet = client.open("Attandance").sheet1  # Replace "Classroom Attendance" with your sheet name

# Initialize Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds for blink detection
EYE_AR_THRESHOLD = 0.2  # Eyes closed threshold
BLINK_FRAMES = 3  # Number of consecutive frames required for a single blink
REQUIRED_BLINKS = 2  # Number of successful blinks needed to mark attendance

blink_counter = {}
blink_count = {}
blink_detected = {}

# Cached student names (to minimize API calls)
cached_student_names = []

# Open webcam
video_capture = cv2.VideoCapture(0)

# Cache the student names from the Google Sheet only once
def cache_student_names():
    global cached_student_names
    existing_data = attendance_sheet.get_all_records()  # Fetch all the records
    cached_student_names = [entry['Name'] for entry in existing_data]
    print(f"Cached student names: {cached_student_names}")

# Check if the student's name is in the cached list
def is_student_present(name):
    return name in cached_student_names

# Cache the data initially
cache_student_names()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break  # Exit if no frame is captured

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

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Initialize blink counters if not already set
            if name not in blink_counter:
                blink_counter[name] = 0
                blink_count[name] = 0
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
                    blink_count[name] += 1  # Increment blink count
                    print(f"Blink {blink_count[name]} detected for {name} ✅")

                blink_counter[name] = 0  # Reset counter when eyes open

            # Before marking attendance, check if the student is already in the cached list
            if not is_student_present(name):
                # Mark attendance if student hasn't marked attendance yet
                if blink_count[name] >= REQUIRED_BLINKS:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance_sheet.append_row([name, current_time])  # Ensure both Name and Time are added
                    cached_student_names.append(name)  # Add student name to the cached list
                    print(f"✅ Attendance marked for {name} at {current_time}")
                    blink_count[name] = 0  # Reset blink count for next detection
            else:
                print(f"Attendance already marked for {name}. Skipping...")

        # Draw a rectangle and label on the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Attendance System", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Introduce a small delay to avoid exceeding the API quota
    time.sleep(0.2)

# Release resources
video_capture.release()
cv2.destroyAllWindows()
print("Attendance system closed.")
