from pathlib import Path
import cv2

import face_recognition, pickle


DEFAULT_ENCODINGS_PATH = Path("model/encodings.pkl")
OUTPUT_FRAMES_PATH = Path("output/frames")
MOV_FILE_PATH = "training/1.mov"

# Ensure necessary directories exist
OUTPUT_FRAMES_PATH.mkdir(parents=True, exist_ok=True)

def encode_faces_from_video(
    video_path: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    """
    Encodes faces from a video file and saves the encodings for later use.
    """
    names = []       # List to hold the names (e.g., "My Face")
    encodings = []   # List to hold the face encodings

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    processed_frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video

        frame_count += 1
        # Process every nth frame to reduce processing time
        if frame_count % 10 != 0:  # Skip frames to save time
            continue

        # Convert frame to RGB (face_recognition expects RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Save each encoding and associate with a name
        for encoding in face_encodings:
            names.append("Alex")  # Label for all encodings in the video
            encodings.append(encoding)

        # Optional: Save the frame for debugging
        frame_path = OUTPUT_FRAMES_PATH / f"frame_{frame_count}.jpg"
        cv2.imwrite(str(frame_path), frame)

        processed_frame_count += 1
        print(f"Processed frame {frame_count}, detected {len(face_encodings)} face(s)")

    # Save the encodings to a file
    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

    video_capture.release()
    print(f"Processing complete. Encodings saved to {encodings_location}")
    print(f"Total frames processed: {processed_frame_count}")

# Call the function to encode faces from your video
encode_faces_from_video(MOV_FILE_PATH)