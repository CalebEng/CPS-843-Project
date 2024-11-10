import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import random

# Load the trained model
model = load_model('newmodel.h5')

# Define emotion labels (assuming the same order as in training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to test random images from each emotion category in the test set
def test_random_images(test_dir):
    plt.figure(figsize=(10, 10))
    class_labels = emotion_labels
    for i, emotion_class in enumerate(class_labels):
        class_dir = os.path.join(test_dir, emotion_class)
        random_image = random.choice(os.listdir(class_dir))
        img_path = os.path.join(class_dir, random_image)
        
        # Load and preprocess the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (48, 48))
        img_normalized = img_resized / 255.0
        img_normalized = np.reshape(img_normalized, (1, 48, 48, 1))
        
        # Predict emotion
        prediction = model.predict(img_normalized)
        predicted_label = class_labels[np.argmax(prediction)]
        
        # Display the image and prediction
        plt.subplot(3, 3, i + 1)
        plt.imshow(img_resized, cmap='gray')
        plt.title(f"True: {emotion_class}\nPred: {predicted_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Run the function to display random images from the test set
test_dir = 'data/test'  # Define the path to your test data directory
test_random_images(test_dir)

# Initialize the webcam for real-time emotion detection
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Matplotlib for probability display
plt.ion()
fig, ax = plt.subplots()
bars = ax.bar(emotion_labels, [0]*7)
ax.set_ylim([0, 1])
ax.set_title("Emotion Probabilities")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the face from the grayscale frame
        face_gray = gray[y:y+h, x:x+w]
        
        # Resize the face to 48x48 pixels (the input size for our model)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_normalized = face_resized / 255.0  # Normalize pixel values
        face_normalized = np.reshape(face_normalized, (1, 48, 48, 1))  # Reshape to match model input

        # Predict the emotion
        prediction = model.predict(face_normalized)
        emotion_label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Display the label and confidence
        label = f"{emotion_label} ({confidence * 100:.2f}%)"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Update the probability bar chart
        for i, bar in enumerate(bars):
            bar.set_height(prediction[0][i])
        plt.draw()
        plt.pause(0.001)

    # Display the frame with emotion labels
    cv2.imshow('Emotion Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
plt.close()
