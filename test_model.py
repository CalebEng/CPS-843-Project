import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import random

model = load_model('63percentaccuracy.h5')

# define emotion labels 
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# use this to test our model on randomly selected images in our test dataset
def test_random_images(test_dir):
    plt.figure(figsize=(10, 10))
    class_labels = emotion_labels
    for i, emotion_class in enumerate(class_labels):
        # iterate through each emotion folder in the test data set and generate a random image
        class_dir = os.path.join(test_dir, emotion_class)
        random_image = random.choice(os.listdir(class_dir))
        img_path = os.path.join(class_dir, random_image)
        
        # once the image is selected, preprocess the data
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (48, 48))
        img_normalized = img_resized / 255.0
        img_normalized = np.reshape(img_normalized, (1, 48, 48, 1))
        
        prediction = model.predict(img_normalized)
        predicted_label = class_labels[np.argmax(prediction)]
        
        # predict emotion
        plt.subplot(3, 3, i + 1)
        plt.imshow(img_resized, cmap='gray')
        plt.title(f"True: {emotion_class}\nPred: {predicted_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# define test data file path
test_dir = 'data/test' 
test_random_images(test_dir)

# invoke camera for live video captrue
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# matplotlib for probability bar chart
plt.ion()
fig, ax = plt.subplots()
bars = ax.bar(emotion_labels, [0]*7)
ax.set_ylim([0, 1])
ax.set_title("Emotion Probabilities")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # extract the face from the grayscale frame
        face_gray = gray[y:y+h, x:x+w]
        
        # resize the face to base input size of 48x48 pixels
        face_resized = cv2.resize(face_gray, (48, 48))
        face_normalized = face_resized / 255.0  # Normalize pixel values
        face_normalized = np.reshape(face_normalized, (1, 48, 48, 1))  # Reshape to match model input

        # predict emotion by passing normalized face into our model
        prediction = model.predict(face_normalized)
        emotion_label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # display the predicted output and confidence
        label = f"{emotion_label} ({confidence * 100:.2f}%)"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # invoke update on bar chart
        for i, bar in enumerate(bars):
            bar.set_height(prediction[0][i])
        plt.draw()
        plt.pause(0.001)

    cv2.imshow('Emotion Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close()
