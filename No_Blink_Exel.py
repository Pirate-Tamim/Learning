import face_recognition
import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime

# Load known faces
with open("face_encodings.pkl", "rb") as file:
    known_face_encodings, known_face_names = pickle.load(file)

# Initialize attendance file
attendance_file = "attendance.csv"

# Load existing attendance file or create a new one
try:
    df = pd.read_csv(attendance_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Name", "Time"])

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, frame = video_capture.read()
    if not ret:
        break  # Exit if no frame is captured

    # Convert the frame to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through detected faces
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Mark attendance only if not already recorded for today
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if name not in df["Name"].values:
                new_entry = pd.DataFrame([[name, current_time]], columns=["Name", "Time"])
                df = pd.concat([df, new_entry], ignore_index=True)
                df.to_csv(attendance_file, index=False)
                print(f"✅ Attendance marked for {name} at {current_time}")

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
