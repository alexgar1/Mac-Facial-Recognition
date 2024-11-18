
import pickle, os, cv2, face_recognition
from pathlib import Path
from PIL import Image, ImageDraw

# Path to the encodings file
ENCODINGS_PATH = Path("model/encodings.pkl")

# Load pre-trained face encodings
def load_encodings(encodings_location: Path):
    with encodings_location.open(mode="rb") as f:
        return pickle.load(f)

loaded_encodings = load_encodings(ENCODINGS_PATH)

def recognize_face(unknown_encoding, known_encodings):
    """
    Compares an unknown face encoding to known encodings.
    Returns the name of the recognized person or None if not matched.
    """
    names = known_encodings["names"]
    encodings = known_encodings["encodings"]
    matches = face_recognition.compare_faces(encodings, unknown_encoding)
    if True in matches:
        match_index = matches.index(True)
        return names[match_index]
    return None

def draw_bounding_box(frame, bounding_box, name):
    """
    Draws a bounding box and name on the frame.
    """
    top, right, bottom, left = bounding_box
    # Draw a rectangle around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    # Add name label below the rectangle
    cv2.putText(
        frame,
        name,
        (left, bottom + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
    )

# Start the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for bounding_box, unknown_encoding in zip(face_locations, face_encodings):
        # Recognize the face
        name = recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        # Draw the bounding box and name on the frame
        draw_bounding_box(frame, bounding_box, name)

        if name != 'Alex':
            os.system('say gypsy')
        else:
            os.system('say good boy')


    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
