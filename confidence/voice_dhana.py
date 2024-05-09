import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import dlib

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You need to download this file

# Load pre-trained CNN model for facial expression recognition
cnn_model = load_model('emotion.h5')  # You need to train or obtain this model
# Load pre-trained CNN model for facial expression recognition


# Load the video
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video file. Please check the file path.")
    exit()

# Initialize variables
confidence_scores = []

# Function to detect facial landmarks using dlib
def detect_landmarks(image, face):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, face)
    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
    return landmarks

# Function to preprocess the face image for the CNN model
def preprocess_for_cnn(image, face):
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_img = image[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (224, 224))  # Assuming the CNN model expects input size of 224x224
    face_img = preprocess_input(face_img)
    return face_img

# Function to calculate confidence score based on expression prediction
def calculate_confidence_score(expression_prediction):
    # You need to define how to interpret the prediction probabilities to calculate confidence score
    # This could involve thresholding or using a weighted combination of probabilities
    confidence_score = expression_prediction  # Placeholder, replace this with your calculation
    return confidence_score

# Process video frame by frame
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Check if faces are detected
    if len(faces) == 0:
        # Handle case where no faces are detected
        continue

    for face in faces:
        # Extract facial landmarks using dlib
        landmarks = detect_landmarks(frame, face)
        
        # Preprocess the frame for the CNN model
        face_img = preprocess_for_cnn(frame, face)
        
        # Predict facial expression using the CNN model
        expression_prediction = cnn_model.predict(np.expand_dims(face_img, axis=0))
        print(expression_prediction)
        
        # Calculate confidence score based on expression prediction
        confidence_score = calculate_confidence_score(expression_prediction)
        confidence_scores.append(confidence_score)

# Check if confidence_scores list is empty
if len(confidence_scores) == 0:
    print("No faces detected or invalid model predictions.")
else:
    # Analyze confidence scores
    average_confidence = np.mean(confidence_scores)
    print("Average confidence score:", average_confidence)

# Release the video capture object
cap.release()