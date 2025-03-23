import os
import cv2
import numpy as np
import pandas as pd

def load_data(data_dir, labels_path):
    try:
        # Load labels from CSV
        labels_df = pd.read_csv(labels_path)
        images = []
        labels = []

        # Load and preprocess images
        for idx, row in labels_df.iterrows():
            image_path = os.path.join(data_dir, row['image_path'])
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                continue  # Skip missing images

            label = row['species']
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read image at {image_path}")
                continue  # Skip corrupted images

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
