import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model('fault_classifier.keras')

# Path to the folder containing images
folder_path = "Scalogram_Combined_2\\Normal"

# List all files in the folder
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png'))]

#label = []

for image_file in image_files:
    # Read and preprocess the image
    img_path = os.path.join(folder_path, image_file)
    img = Image.open(img_path)
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    prediction = model.predict(img_array)
    class_label = np.argmax(prediction)
    #label.append(class_label)
    print(f"Predicted class: {class_label}")
    