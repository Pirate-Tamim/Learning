import pickle

# Load encodings
with open("face_encodings.pkl", "rb") as file:
    known_face_encodings, known_face_names = pickle.load(file)

print("Encodings Loaded Successfully!")
print(known_face_encodings)
print(known_face_names)
