import cv2

# Load pre-trained models with full paths
face_cascade = cv2.CascadeClassifier('D:/C_files/Python/Confidence/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('D:/C_files/Python/Confidence/haarcascade_eye.xml')

# Load the video
cap = cv2.VideoCapture("video.mp4")

# Initialize variables for overall confidence calculation
total_confidence = 0
num_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Increment the number of frames processed
    num_frames += 1

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Calculate confidence based on the number of detected eyes
    for (x, y, w, h) in faces:
        # Extract region of interest (ROI) containing the face
        face_roi = gray[y:y+h, x:x+w]

        # Detect eyes in the face ROI
        eyes = eye_cascade.detectMultiScale(face_roi)
        
        # Calculate confidence based on the number of detected eyes
        num_eyes = len(eyes)
        confidence = min(num_eyes / 2, 1.0)  # Cap confidence at 1.0 (100%)
        
        # Accumulate confidence for overall calculation
        total_confidence += confidence

# Calculate the overall confidence
overall_confidence = total_confidence / num_frames if num_frames > 0 else 0

# Print the overall confidence
print("Overall Confidence:", round(overall_confidence * 100, 2), "%")

# Release the video capture object and close the OpenCV windows
cap.release()

