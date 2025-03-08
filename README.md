# 🧠 Brain Tumor Classification Using CNN  

This project utilizes **Convolutional Neural Networks (CNNs)** for **brain tumor detection** using MRI images. The model classifies images into **four categories**: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**. The goal is to assist in **early diagnosis** and improve **medical decision-making** using AI-powered deep learning techniques.  

---

## 📌 Table of Contents  
- [Introduction](#introduction)  
- [Problem Statement](#problem-statement)  
- [Importance of This Project](#importance-of-this-project)  
- [Methodology](#methodology)  
- [Model Architecture](#model-architecture)  
- [Training Process](#training-process)  
- [Results and Performance](#results-and-performance)  
- [Conclusion](#conclusion)  

---

## 🏥 Introduction  
Brain tumors are **life-threatening conditions** requiring **early and accurate diagnosis**. Manual MRI analysis is **time-consuming** and **prone to errors**. This project **automates tumor classification** using **deep learning**, reducing the **diagnostic burden** on radiologists.  

---

## ❌ Problem Statement  
Traditional methods for tumor diagnosis have limitations:  
- **Subjectivity**: Human errors in diagnosis  
- **Time Constraints**: Manual MRI analysis takes too long  
- **Inconsistencies**: Variability in radiologists' assessments  

This project tackles these challenges by implementing **CNN-based image classification**.  

---

## 🧬 Importance of This Project  
✅ **Early Detection**: Faster and more accurate **tumor diagnosis**  
✅ **Reduced Human Effort**: AI-powered automation for **quick decision-making**  
✅ **Improved Accuracy**: CNN-based classification ensures **high reliability**  

---

## ⚙️ Methodology  

### 📌 Data Preparation  
- **Resizing**: Images scaled to `150×150` for consistency  
- **Normalization**: Pixel values scaled to `[0,1]` for training efficiency  
- **Augmentation**: Applied techniques like **flipping, rotation, scaling, shearing, mixup, and mosaic**  
- **Caching**: Storing images in memory to speed up training  
- **Class Labels**: The dataset includes four categories:  
  - 🧠 **Glioma Tumor**  
  - 🏥 **Meningioma Tumor**  
  - ✅ **No Tumor (Healthy Brain)**  
  - 🧬 **Pituitary Tumor**

---
## 🏗 Model Architecture  

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

### 📌 Training Process  

| Parameter         | Value                     |  
|------------------|-------------------------|  
| **Model**        | CNN                       |  
| **Epochs**       | 20                        |  
| **Batch Size**   | 16                        |  
| **Image Size**   | 150×150                   |  
| **Optimizer**    | Adam                      |  
| **Learning Rate** | Adaptive (starting at 0.001) |  
| **Loss Function** | Categorical Crossentropy  |  

### Results and Performance
✅ Loss Analysis
📉 Training & Validation Loss steadily decreased, proving effective learning.

✅ Precision & Recall
✔ Precision improved consistently, reaching 92%.
✔ Recall showed a steady increase, reducing misclassification.

✅ Confusion Matrix & Classification Report

### Conclusion
The CNN model successfully classifies brain tumors with high accuracy.
AI-based tumor detection can assist radiologists and speed up diagnosis.
The project proves deep learning is effective in medical image analysis.

