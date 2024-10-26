import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

from utils import get_face_landmarks

# Define emotion labels
emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Load the trained CNN model
model = load_model('emotion_recognition_cnn.h5')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

def draw_probabilities(probabilities):
    # Create a black image for the probability graph (width 400, height 300)
    graph = np.zeros((300, 400, 3), dtype=np.uint8)

    # Set the font and other display parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    bar_width = 50
    max_bar_height = 200  # Max height of the bars in the graph
    
    # Loop through the probabilities and draw bars
    for i, prob in enumerate(probabilities):
        # Calculate the bar height based on probability
        bar_height = int(prob * max_bar_height)
        x = i * bar_width + 10
        
        # Draw a filled rectangle representing the bar
        cv2.rectangle(graph, (x, 250 - bar_height), (x + bar_width - 10, 250), (0, 255, 0), -1)
        
        # Display the emotion label below each bar
        cv2.putText(graph, emotions[i], (x, 270), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display the probability percentage above each bar
        prob_text = f"{prob * 100:.1f}%"
        cv2.putText(graph, prob_text, (x, 240 - bar_height), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return graph

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Extract face landmarks from the current frame
    face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)
    
    if len(face_landmarks) == 1404:  # Ensure the expected number of landmarks
        # Reshape and normalize the landmarks to match the model's input
        face_landmarks = np.array(face_landmarks).reshape(1, 26, 54, 1)  # Reshape for CNN
        face_landmarks = face_landmarks / np.max(face_landmarks)  # Normalize

        # Predict the emotion probabilities
        probabilities = model.predict(face_landmarks)[0]  # Get the first (and only) sample's probabilities
        emotion_index = np.argmax(probabilities)
        emotion_label = emotions[emotion_index]

        # Display the emotion label on the frame
        cv2.putText(frame, emotion_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Generate the probability graph
        graph = draw_probabilities(probabilities)

        # Show the probability graph in a separate window
        cv2.imshow('Emotion Probabilities', graph)

    # Show the webcam feed with the predicted emotion
    cv2.imshow('Emotion Recognition', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
