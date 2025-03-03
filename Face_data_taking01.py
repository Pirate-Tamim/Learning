import cv2
import os

# Create a directory to store all student images
base_dir = "student_images"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

def capture_student_images(student_name, num_photos=10):
    """
    Captures multiple images for a student and saves them in their respective folder.
    
    Args:
        student_name (str): The name of the student.
        num_photos (int): Number of photos to capture.
    """
    # Create a folder for the student
    student_dir = os.path.join(base_dir, student_name)
    if not os.path.exists(student_dir):
        os.makedirs(student_dir)

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Capturing images for {student_name}. Press 'c' to capture or 'q' to quit early.")
    photo_count = 0

    while photo_count < num_photos:
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to capture frame. Please try again.")
            break

        # Display the video feed
        cv2.imshow(f"Capturing Photos for {student_name}", frame)

        key = cv2.waitKey(1)
        if key == ord('c'):  # Capture the photo on 'c' press
            photo_path = os.path.join(student_dir, f"{student_name}_{photo_count + 1}.jpg")
            cv2.imwrite(photo_path, frame)
            print(f"Image {photo_count + 1} saved as '{photo_path}'")
            photo_count += 1

        elif key == ord('q'):  # Quit early if 'q' is pressed
            print("Capture process canceled.")
            break

    # Release webcam and close window
    video_capture.release()
    cv2.destroyAllWindows()

# Ask for student's name and capture images
student_name = input("Enter the student's name: ")
num_photos = int(input("Enter the number of photos to capture: "))
capture_student_images(student_name, num_photos)
