# Chest X-Ray Image Classification Using Deep Learning

## Overview

This project explores the classification of chest X-ray images into three medical categories using Deep Learning and Machine Learning techniques. The goal is to compare different classification approaches, evaluate their performance, and identify the most effective architecture for small-scale medical imaging datasets.

## Dataset

The dataset consists of chest X-ray images belonging to three classes. Due to class imbalance and limited data availability, data augmentation techniques were applied to improve model generalization.

## Preprocessing

* Image resizing and standardization
* Grayscale conversion for custom CNN models
* RGB conversion for transfer learning models
* Pixel normalization to the range [0, 1]
* Data augmentation:

  * Rotation
  * Width shift
  * Height shift
  * Zoom
  * Horizontal flip

## Implemented Models

### 1. CNN + Dense Classifier

A custom convolutional neural network consisting of:

* Conv2D(32) + MaxPooling
* Conv2D(64) + MaxPooling
* Conv2D(128) + MaxPooling
* Dense(128) + Dropout(0.5)
* Softmax output layer

### 2. CNN Feature Extraction + Machine Learning Classifiers

CNN feature maps were extracted and used as input to:

* Support Vector Machine (SVM)
* Logistic Regression (LR)
* Random Forest (RF)
* XGBoost (XGB)
* Ensemble model combining classifier probabilities

### 3. Transfer Learning (VGG19)

* Pretrained VGG19 (ImageNet weights)
* Frozen convolutional backbone
* Dense(256) + Dropout(0.5)
* Softmax output layer

## Training

* Optimizer: Adam
* Loss: Sparse Categorical Crossentropy
* Early Stopping
* Class Weight Balancing (for transfer learning model)

## Results

| Model                     | F1 Score |
| ------------------------- | -------- |
| CNN + Dense               | 0.96     |
| CNN + Random Forest       | 0.92     |
| CNN + XGBoost             | 0.92     |
| CNN + SVM                 | 0.91     |
| CNN + Logistic Regression | 0.91     |
| VGG19 Transfer Learning   | 0.94     |

Data augmentation improved CNN performance from approximately 92% to 97% accuracy.

## Technologies

* Python
* TensorFlow / Keras
* Scikit-Learn
* XGBoost
* NumPy
* OpenCV
* Google Colab

## Future Work

* Pathological status detection
* Additional transfer learning architectures (ResNet, DenseNet)
* Advanced image augmentation and synthetic data generation
* Explainable AI techniques for medical decision support

## Authors

* Tomer Peretz
* Lior Cohen
