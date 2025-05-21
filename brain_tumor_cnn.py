#@title
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/content/drive/MyDrive/archive (1)'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf

X_train = []
Y_train = []
image_size = 150
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
for i in labels:
    folderPath = os.path.join('/content/drive/MyDrive/archive (1)/Training', i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        Y_train.append(i)

for i in labels:
    folderPath = os.path.join('/content/drive/MyDrive/archive (1)/Testing', i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        Y_train.append(i)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train,Y_train = shuffle(X_train,Y_train,random_state=101)
X_train.shape
X_train,X_test,y_train,y_test = train_test_split(X_train,Y_train,test_size=0.1,random_state=101)

y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train=y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test=y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import os

# Path to save and load the model
MODEL_PATH = 'saved_model/my_cnn_model.h5'

# Check if model already exists
if os.path.exists(MODEL_PATH):
    print("Loading pre-trained model...")
    model = load_model(MODEL_PATH)
else:
    print("Training new model...")
    # Define model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # Train and save model
    history = model.fit(X_train, y_train, epochs=20, validation_split=0.1)
    model.save(MODEL_PATH)
    print("Model saved to:", MODEL_PATH)

# Show model summary
model.summary()

# Save entire model
model.save('braintumor_model.h5')

# Save only weights
model.save_weights('braintumor_weights.weights.h5')

from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load full model
model = keras.models.load_model('braintumor_model.h5')
# ----------------------------
# Predict on a single image
# ----------------------------
img_path = '/content/drive/MyDrive/archive (1)/Testing/glioma/Te-glTr_0000.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (150, 150))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_array = np.expand_dims(img_rgb, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
plt.imshow(img_rgb)
plt.title(f"Predicted: {labels[predicted_class]}")
plt.axis('off')
plt.show()

print("Predicted class:", labels[predicted_class])

# ----------------------------
# Evaluate on test set
# ----------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:\n", classification_report(y_true_classes, y_pred_classes, target_names=labels))

# Sensitivity, Specificity, Accuracy
TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
TN = np.sum(cm) - (FP + FN + TP)

specificity = np.mean(np.divide(TN, TN + FP, out=np.zeros_like(TN, dtype=float), where=(TN + FP)!=0))
sensitivity = np.mean(np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN)!=0))
accuracy = np.sum(TP) / np.sum(cm)

print("Macro-Averaged Specificity:", round(specificity, 4))
print("Macro-Averaged Sensitivity (Recall):", round(sensitivity, 4))
print("Accuracy:", round(accuracy, 4))
