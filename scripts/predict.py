from tensorflow.keras.models import load_model
import cv2
import numpy as np

def predict_image(model, image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    return predicted_class

if __name__ == "__main__":
    # Load the trained model
    model = load_model('models/cow_model.h5')

    # Path to the test image
    image_path = 'data/raw_images/test_cow.jpg'  # Replace with your test image path

    # Make prediction
    predicted_class = predict_image(model, image_path)
    print(f"Predicted Class: {'cow' if predicted_class == 0 else 'unknown'}")
