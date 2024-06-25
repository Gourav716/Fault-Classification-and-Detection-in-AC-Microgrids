from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import os

# Load the saved model
model = load_model('fault_classifier_3.keras')

img = Image.open("C:\Mark\Major Project\Scalogram_Matlab\CGAN_Fault\Scalogram_Label_2\\Normal\Three_PV_Fault_IF_02.png")

img_width, img_height = 256, 256

# Preprocess the image
img = img.resize((img_width, img_height))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict the class
prediction = model.predict(img_array)

# Get the class label with the highest probability
class_label = np.argmax(prediction)

# Print the predicted class
print(f"Predicted class: {class_label}")
model.summary()

# Save model image to folder
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

'''
# Plot bar graph of classes
image_folder = ['HiF', 'HiFPP', 'P-G', 'P-P']
nimgs = {}
for i in image_folder:
    nimages = len(os.listdir('Scalogram New_2/'+i+'/'))
    nimgs[i]=nimages
plt.figure(figsize=(9, 6))
plt.bar(range(len(nimgs)), list(nimgs.values()), align='center')
plt.xticks(range(len(nimgs)), list(nimgs.keys()))
plt.show()
'''