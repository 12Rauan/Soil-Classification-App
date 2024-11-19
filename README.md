# Soil Classification App

This project is a machine learning-based web application for soil type classification using images. It utilizes a fine-tuned ResNet50 model to classify soil into four categories: **Black Soil**, **Cinder Soil**, **Laterite Soil**, and **Yellow Soil**.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dataset](#dataset)
7. [Model Architecture](#model-architecture)
8. [Results](#results)
9. [Future Improvements](#future-improvements)

---

## Introduction
Soil classification plays a crucial role in agricultural planning and soil analysis. This app simplifies the process by using a deep learning model to classify soil images with high accuracy. Users can upload soil images and receive real-time predictions.

---

## Methodology
1. **Dataset Preparation**:
   - A dataset of soil images was divided into training, validation, and test sets.
   - Data augmentation (e.g., rotation, flipping, resizing) was applied to improve model generalization.

2. **Model Training**:
   - The ResNet50 model was fine-tuned:
     - The last 20 layers of ResNet were unfrozen for retraining.
     - Fully connected layers were modified to predict 4 soil classes.

3. **Evaluation**:
   - The model was optimized using the Adam optimizer and cross-entropy loss.
   - Validation and test datasets were used to measure performance.

4. **Deployment**:
   - The model is integrated with Streamlit for a user-friendly interface, allowing predictions via a simple web application.

---

## Features
- Upload soil images for classification.
- Real-time predictions for soil type.
- Lightweight and easy to deploy locally or on cloud platforms.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/soil-classification-app.git
   cd soil-classification-app

---

## Usage

1. **Start the Application**:
   Run the following command in your terminal to start the Streamlit app:
   ```bash
   streamlit run app.py
2. Open the app in your browser.
3. Upload a soil image in .jpg, .jpeg, or .png format.
4. View the predicted soil type along with a confidence score.

---

## Dataset
The dataset used for training and evaluation contains images of four soil types:
   * Black Soil
   * Cinder Soil
   * Laterite Soil
   *  Yellow Soil
To use this app with your own dataset, replace the existing images and labels in the data directory with your dataset.

---

## Model Architecture

The model is based on the **ResNet50 architecture**:

- **Feature extractor**: Pretrained ResNet50.
- **Classifier head**:
  - Fully connected layers:
    - `512 → BatchNorm → ReLU → Dropout → 256 → BatchNorm → ReLU → Dropout → 4 output classes`.

The final model is trained to classify soil types based on the extracted features.

---

## Results

The model achieved the following performance:

- **Training Accuracy**: 98%
- **Validation Accuracy**: 96%
- **Test Accuracy**: 95%

The app provides reliable predictions for real-world soil classification tasks.

---

## Future Improvements

- Expand the dataset to include additional soil types for broader applicability.
- Deploy the app on cloud platforms such as AWS, Google Cloud, or Heroku.
- Add confidence scores and explainability features (e.g., Grad-CAM) to enhance user trust.
- Improve the interface with advanced visualization tools for better user experience.
