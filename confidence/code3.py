import cv2

# Load pre-trained face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load video file
video_capture = cv2.VideoCapture('video.mp4')

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
video_capture.release()
cv2.destroyAllWindows()

import cv2
import dlib

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You need to download this file

# Load video file
video_capture = cv2.VideoCapture('video_file.mp4')

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = detector(gray)
    
    # Loop over detected faces
    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(gray, face)
        
        # Loop over the facial landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            
            # Draw facial landmarks on the frame
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
video_capture.release()
cv2.destroyAllWindows()
