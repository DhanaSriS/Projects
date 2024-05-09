import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Assuming FER2013 images are stored in directories based on emotion labels
fer2013_dir = 'path_to_fer2013_dataset'

# Define emotions
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Function to load images from directory
def load_images_from_directory(directory):
    images = []
    labels = []
    for emotion_label in emotion_labels:
        emotion_dir = os.path.join(directory, emotion_label)
        for filename in os.listdir(emotion_dir):
            img = cv2.imread(os.path.join(emotion_dir, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))  # Resize images to a fixed size
            images.append(img)
            labels.append(emotion_labels.index(emotion_label))
    return np.array(images), np.array(labels)

# Load images and labels
X_facial, y_facial = load_images_from_directory(fer2013_dir)

# Normalize pixel values
X_facial = X_facial / 255.0

# Split the dataset into training and testing sets
X_facial_train, X_facial_test, y_facial_train, y_facial_test = train_test_split(X_facial, y_facial, test_size=0.2, random_state=42)
