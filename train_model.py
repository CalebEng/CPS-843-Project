import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load data from the text file
data_file = "data.txt"
data = np.loadtxt(data_file)

# Split data into features (X) and labels (y)
X = data[:, :-1]  # Features (landmarks)
y = data[:, -1]   # Labels (emotions)

# Reshape X to fit CNN input (assuming 1404 landmarks; reshape into a suitable shape)
# Example: reshaping into a 2D format, e.g., 26x54, or adjust based on actual data shape
X = X.reshape(-1, 26, 54, 1)  # Reshape to 26x54 with 1 channel (grayscale)

# Normalize data to range [0, 1]
X = X / np.max(X)

# One-hot encode the labels
y = tf.keras.utils.to_categorical(y, num_classes=7)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Build the CNN model
def build_cnn_model():
    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(26, 54, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third convolutional layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Prevent overfitting

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer (7 emotions)
    model.add(Dense(7, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the CNN model
cnn_model = build_cnn_model()

# Train the model
history = cnn_model.fit(X_train, y_train, epochs=70, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the test data
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate classification report and confusion matrix
y_pred = cnn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']))

# Save the trained model
cnn_model.save('emotion_recognition_cnn.h5')
