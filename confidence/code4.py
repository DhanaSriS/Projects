import cv2
import dlib

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You need to download this file

# Load video file
video_capture = cv2.VideoCapture('video.mp4')

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
