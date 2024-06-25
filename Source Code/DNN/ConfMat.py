from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array

img_width, img_height = 256, 256

def preprocess_image(path):
    img = load_img(path, target_size = (img_height, img_width))
    a = img_to_array(img)
    a = np.expand_dims(a, axis = 0)
    a /= 255.
    return a

# Load the saved model
model = load_model("fault_classifier.keras")

test_images_dir = 'Scalogram_2/'
df = pd.read_csv('Scalogram_Class_2.csv')

# put them in a list
test_dfToList = df['id'].tolist()
test_ids = [str(item) for item in test_dfToList]

test_images = [test_images_dir+item for item in test_ids]
test_preprocessed_images = np.vstack([preprocess_image(fn) for fn in test_images])
#np.save('test_preproc_CNN.npy', test_preprocessed_images)

# Predict the class
prediction = model.predict(test_preprocessed_images)

# Get the class label with the highest probability
class_label = np.argmax(prediction, axis = 1)

# Print the predicted class
print(f"Class Label: {class_label}")

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

y_true = df['class']
y_pred = prediction

from sklearn.metrics import log_loss
loss = log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)

conf_mat = confusion_matrix(y_true, class_label)
print(conf_mat)

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(2)
        #print("Normalized confusion matrix")
    else:
        cm=cm
        #print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

classes = os.listdir('Scalogram_Label_2')
np.set_printoptions(precision=2)
fig1 = plt.figure(figsize=(7,6))
plot_confusion_matrix(conf_mat, classes=classes, title='Confusion Matrix')
plt.show()

print(classification_report(y_true, class_label))