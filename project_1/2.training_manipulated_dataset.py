import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


class ImageClassifier:
    def __init__(self, data_dir='dataset/TrainImages'):
        self.data_dir = data_dir
        self.image_exts = ['jpeg', 'jpg', 'bmp', 'png', 'pgm']
        self.train = None
        self.val = None
        self.test = None
        self.models = {}
        self.results = {}

    def setup_gpu(self):
        """Configures GPU memory growth if GPUs are available."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPUs found and configured: {len(gpus)}")
            except RuntimeError as e:
                print(f"Error configuring GPUs: {e}")

    def load_data(self):
        """Loads images and labels from 'pos' and 'neg' folders."""
        pos_dir = os.path.join(self.data_dir, 'pos')
        neg_dir = os.path.join(self.data_dir, 'neg')

        if not os.path.exists(pos_dir) or not os.path.exists(neg_dir):
            raise FileNotFoundError("The dataset folder structure is invalid. Ensure 'pos' and 'neg' folders exist.")

        def load_images_from_folder(folder, label):
            images, labels = [], []
            for filename in os.listdir(folder):
                ext = filename.split('.')[-1].lower()
                if ext in self.image_exts:
                    filepath = os.path.join(folder, filename)
                    try:
                        img = load_img(filepath, color_mode='grayscale', target_size=(40, 100))
                        img_array = img_to_array(img) / 255.0  # Normalize
                        images.append(img_array)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {filename}: {e}")
            return images, labels

        # Load images from both positive and negative folders
        pos_images, pos_labels = load_images_from_folder(pos_dir, 1)  # Label 1 for cars
        neg_images, neg_labels = load_images_from_folder(neg_dir, 0)  # Label 0 for no cars

        images = np.array(pos_images + neg_images)
        labels = np.array(pos_labels + neg_labels)

        # Shuffle and batch the dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(buffer_size=len(labels)).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.data = dataset

        # Check data distribution
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        print(f"Data distribution: {distribution}")

    def split_data(self):
        """Splits the dataset into training, validation, and test sets."""
        cardinality = tf.data.experimental.cardinality(self.data).numpy()
        train_size = int(cardinality * 0.6)
        val_size = int(cardinality * 0.2)

        self.train = self.data.take(train_size)
        self.val = self.data.skip(train_size).take(val_size)
        self.test = self.data.skip(train_size + val_size)

    def build_profile_view_cnn(self):
        """Builds a CNN architecture for input size 40x100."""
        model = Sequential([
            Conv2D(32, (5, 5), activation='relu', input_shape=(40, 100, 1)),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(32, (5, 5), activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(64, (5, 5), activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(1024, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.models['Profile-View CNN'] = model

    def train_all_models(self, epochs=10):
        """Trains all models and stores their training history."""
        for name, model in self.models.items():
            print(f"Training {name}...")
            history = model.fit(self.train, epochs=epochs, validation_data=self.val, verbose=1)
            self.results[name] = history
            model.save(f"{name.replace(' ', '_').lower()}_model.h5")
            print(f"Model saved as {name.replace(' ', '_').lower()}_model.h5")

    def evaluate_model(self):
        """Evaluates all models on the test set and generates confusion matrices."""
        for name, model in self.models.items():
            test_loss, test_accuracy = model.evaluate(self.test, verbose=1)
            print(f"{name} Test Accuracy: {test_accuracy:.4f}")

            y_true = np.concatenate([y.numpy() for _, y in self.test], axis=0)
            y_pred = np.concatenate([model.predict(x).flatten() for x, _ in self.test])

            y_pred_binary = (y_pred > 0.5).astype(int)
            cm = confusion_matrix(y_true, y_pred_binary)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Car", "Car"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix for {name}")
            plt.show()

    def plot_results(self):
        """Plots training and validation accuracy/loss for all models."""
        plt.figure(figsize=(12, 6))
        for name, history in self.results.items():
            plt.plot(history.history['loss'], label=f'{name} Training Loss')
            plt.plot(history.history['val_loss'], label=f'{name} Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        for name, history in self.results.items():
            plt.plot(history.history['accuracy'], label=f'{name} Training Accuracy')
            plt.plot(history.history['val_accuracy'], label=f'{name} Validation Accuracy')
        plt.title('Training and Validation Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    classifier = ImageClassifier(data_dir='dataset/TrainAugmentedImages')
    # classifier.setup_gpu()
    classifier.load_data()
    classifier.split_data()
    classifier.build_profile_view_cnn()
    classifier.train_all_models(epochs=10)
    classifier.evaluate_model()
    classifier.plot_results()