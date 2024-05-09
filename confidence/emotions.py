import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained emotion classification model
emotion_model = load_model('emotion_detection_model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect faces and predict emotions
def detect_emotions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        resized_roi = cv2.resize(face_roi, (48, 48))
        img = image.img_to_array(resized_roi)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        prediction = emotion_model.predict(img)
        max_index = np.argmax(prediction[0])
        emotion = emotion_labels[max_index]
        
        cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return frame

# Load the video file
video_capture = cv2.VideoCapture('input_video.mp4')

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Detect emotions in the current frame
    frame = detect_emotions(frame)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
