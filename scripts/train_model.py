import os
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint


def load_data(data_dir, labels_path):
    try:
        # Load labels from CSV
        labels_df = pd.read_csv(labels_path)
        print("Labels DataFrame:", labels_df.head())

        images = []
        labels = []

        for idx, row in labels_df.iterrows():
            image_path = os.path.join(data_dir, row['image_path'])
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                continue  # skip missing images

            label = row['species']
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read image at {image_path}")
                continue  # skip corrupted images

            image = cv2.resize(image, (224, 224))
            image = image / 255.0
            images.append(image)
            labels.append(label)

        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        return images, labels

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


# Split the data
def split_data(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


# Build the model
def build_model(num_classes):
    # Load MobileNetV2 (pre-trained on ImageNet)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    # Add custom layers for classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# Train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=10):
    checkpoint = ModelCheckpoint('models/best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[checkpoint])
    return history


# Main function
def main():
    data_dir = 'data/raw_images'
    labels_path = 'data/labels.csv'

    images, labels = load_data(data_dir, labels_path)
    if images is None or labels is None:
        print("Error: Failed to load data. Exiting.")
        sys.exit(1)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    num_classes = len(np.unique(labels))
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, labels)

    # Build model
    model = build_model(num_classes)

    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=10)

    # Save the final model
    model.save('models/cow_model.h5')

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    print("Model trained and saved successfully!")


if __name__ == "__main__":
    main()