import face_recognition
import pickle
import os
import numpy as np

# Root folder containing student images
image_root_folder = "D:\Resources For Projects\Classroom_Attendance_System\student_images"

# Prepare lists to store encodings and names
known_face_encodings = []
known_face_names = []

# Loop through each student folder
for student_folder in os.listdir(image_root_folder):
    student_folder_path = os.path.join(image_root_folder, student_folder)

    # Check if it's a directory
    if os.path.isdir(student_folder_path):
        print(f"Processing images for: {student_folder}")
        
        student_encodings = []  # Temporary list to store all encodings for a student
        
        # Loop through all images in the student's folder
        for filename in os.listdir(student_folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Process only image files
                image_path = os.path.join(student_folder_path, filename)

                # Load image
                image = face_recognition.load_image_file(image_path)

                # Encode face(s)
                face_encodings = face_recognition.face_encodings(image)

                # If faces are found, process the first face
                if face_encodings:
                    student_encodings.append(face_encodings[0])  # Collect all encodings for the student

        # If there are encodings for this student, compute the average and store it
        if student_encodings:
            average_encoding = np.mean(student_encodings, axis=0)  # Compute the average encoding
            known_face_encodings.append(average_encoding)  # Store averaged encoding
            known_face_names.append(student_folder)  # Store the student’s name (folder name)

# Save encodings and names to file
with open("face_encodings.pkl", "wb") as file:
    pickle.dump((known_face_encodings, known_face_names), file)  # Store as a tuple

print("Face encodings for all students saved successfully!")
