import face_recognition
import cv2
from collections import deque

# Load known faces encoding and names
known_faces_encoding = []
known_faces_name = []

# Load the known faces from files
known_person1_image = face_recognition.load_image_file("images/mypic7.jpg")
known_person2_image = face_recognition.load_image_file("images/cr7.jpeg")

known_person1_encoding = face_recognition.face_encodings(known_person1_image)[0]
known_person2_encoding = face_recognition.face_encodings(known_person2_image)[0]

known_faces_encoding.extend([known_person1_encoding, known_person2_encoding])
known_faces_name.extend(["Abdullah", "Cristiano Ronaldo"])

# Start video capture
video_cap = cv2.VideoCapture(0)

# Cache to store face encodings and their locations
face_cache = deque(maxlen=10)  # Store up to 10 frames of face encodings
frame_counter = 0

while True:
    ret, frame = video_cap.read()
    if not ret:
        break

    # Resize the frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)

    # Clear the cache if no faces are detected
    if not face_locations:
        face_cache.clear()

    # List to store names for the current frame
    names = []

    for (top, right, bottom, left) in face_locations:
        # Scale back the face locations to the original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Check if the face is already in the cache
        name = "Unknown"
        for cached_face in face_cache:
            cached_location, cached_encoding, cached_name = cached_face
            # Compare the current face location with cached locations
            if (abs(top - cached_location[0]) < 20 and
                abs(right - cached_location[1]) < 20 and
                abs(bottom - cached_location[2]) < 20 and
                abs(left - cached_location[3]) < 20):
                name = cached_name
                break

        # If the face is not in the cache, encode it
        if name == "Unknown":
            face_encoding = face_recognition.face_encodings(rgb_small_frame, [(top // 4, right // 4, bottom // 4, left // 4)])[0]
            matches = face_recognition.compare_faces(known_faces_encoding, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_faces_name[first_match_index]

            # Add the face to the cache
            face_cache.append(((top, right, bottom, left), face_encoding, name))

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Increment frame counter
    frame_counter += 1

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()