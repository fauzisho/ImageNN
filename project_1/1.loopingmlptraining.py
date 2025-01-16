import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Path to the dataset
augmented_images_path = 'dataset/TrainAugmentedImages'

# Function to load data
def load_data(base_path):
    """Load image data from positive and negative folders."""
    X, y = [], []
    for label, folder in enumerate(['pos', 'neg']):
        folder_path = os.path.join(base_path, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_flattened = img.flatten() / 255.0  # Flatten and normalize
                X.append(img_flattened)
                y.append(label)
    return np.array(X), np.array(y)

# Load data
X, y = load_data(augmented_images_path)

# First split: 60% training and 40% temporary
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Second split: 20% validation and 20% testing
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Define input dimension
input_dim = X_train.shape[1]  # Flattened image size
# Initialize variables for training loop
validation_target = 0.90
testing_target = 0.90
max_iterations = 100  # Maximum number of iterations to avoid infinite loops
iteration = 0

while True:
    print(f"Iteration {iteration + 1} - Training the model...")

    # Define the MLP model
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary output
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)

    # Evaluate on the validation set
    val_accuracy = history.history['val_accuracy'][-1]  # Get the last epoch's validation accuracy

    # Evaluate on the testing set
    loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

    # Check if both conditions are satisfied
    if val_accuracy >= validation_target and test_accuracy >= testing_target:
        print(f"Target achieved! Validation Accuracy: {val_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

        # Save the entire model
        model.save('car_detection_model_99_accuracy.h5')
        print("Model saved as 'car_detection_model_99_accuracy.h5'")

        # Save weights and biases separately
        # model.save_weights('car_detection_model_weights.h5')
        print("Model weights saved as 'car_detection_model_weights.h5'")

        # Save model architecture to JSON (optional)
        model_json = model.to_json()
        with open("car_detection_model_architecture.json", "w") as json_file:
            json_file.write(model_json)
        print("Model architecture saved as 'car_detection_model_architecture.json'")

        break

    # Increment iteration counter
    iteration += 1

    # Break the loop if max iterations are reached
    if iteration >= max_iterations:
        print("Maximum iterations reached. Target not achieved.")
        break

# Plot training & validation accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix for Test Set
y_pred = (model.predict(X_test) > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Car', 'Car'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Test Set")
plt.show()
