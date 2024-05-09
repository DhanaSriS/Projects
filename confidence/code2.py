import cv2
import dlib

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the video
cap = cv2.VideoCapture('input_video.mp4')

detected_faces = []
detected_landmarks = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract region of interest (ROI) for facial landmark detection
        face_gray = gray[y:y+h, x:x+w]
        
        # Detect facial landmarks
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = predictor(gray, dlib_rect)
        
        detected_faces.append((x, y, w, h))
        detected_landmarks.append([(p.x, p.y) for p in landmarks.parts()])

# Calculate confidence based on the number of detected faces and landmarks
face_confidence = len(detected_faces)
landmark_confidence = len(detected_landmarks)

# Overall confidence
overall_confidence = (face_confidence + landmark_confidence) / 2

# Normalize overall confidence to a scale of 0 to 10
max_confidence = 68  # Assuming the maximum number of facial landmarks detected
normalized_confidence = (overall_confidence / max_confidence) * 10

# Ensure confidence is within the range of 0 to 10
normalized_confidence = min(max(normalized_confidence, 0), 10)

print("Total Faces Detected:", face_confidence)
print("Total Facial Landmarks Detected:", landmark_confidence)
print("Overall Confidence (out of 10):", normalized_confidence)

# Release the video capture object
cap.release()
