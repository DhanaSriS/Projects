import cv2
import dlib

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the video
cap = cv2.VideoCapture('input_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Draw bounding boxes around detected faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract region of interest (ROI) for facial landmark detection
        face_gray = gray[y:y+h, x:x+w]
        
        # Detect facial landmarks
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = predictor(gray, dlib_rect)
        
        # Example: Draw points for each facial landmark
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    
    # Calculate confidence based on the number of detected faces and landmarks
    face_confidence = len(faces) / (frame.shape[0] * frame.shape[1])  # Normalize by frame size
    landmark_confidence = len(landmarks.parts()) / 68  # Assuming 68 landmarks are detected

    # Overall confidence
    overall_confidence = (face_confidence + landmark_confidence) / 2

    print("Face Detection Confidence:", face_confidence)
    print("Facial Landmark Detection Confidence:", landmark_confidence)
    print("Overall Confidence:", overall_confidence)
    
    # Display the frame
    cv2.imshow('Face Detection and Landmark Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
