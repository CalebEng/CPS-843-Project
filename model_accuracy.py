from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# load model
model = load_model('63percentaccuracy.h5')

# load test data into ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode='categorical',
    batch_size=64,
    shuffle=False
)

# output model accuracy on test data
loss, accuracy = model.evaluate(test_generator, verbose=0)
print(f"Loaded model accuracy on test data: {accuracy * 100:.2f}%")

Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

# display confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_generator.class_indices.keys(), 
            yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# print classification report
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
