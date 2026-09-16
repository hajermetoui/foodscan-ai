# FoodScan AI — Food Recognition & Calorie Estimation

Image classification over 101 food categories with per-portion calorie estimation, built using transfer learning.

## Overview
A deep-learning model that identifies a meal from a photo (Food-101 dataset, 101 categories, 101k images) and estimates its calorie content. Built with transfer learning on MobileNetV2 and two-phase fine-tuning.

## Approach
- **Transfer learning** with MobileNetV2 (pre-trained on ImageNet)
- **Two-phase training:** train the classification head, then fine-tune the last 30 layers at a reduced learning rate
- **Data pipeline:** normalization, resizing to 224×224, real-time augmentation (flip, zoom, rotation), 80/20 train–validation split
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Returns the top-3 predictions with confidence scores and an estimated calorie value

## Tools
TensorFlow / Keras · MobileNetV2 · Python · Google Colab

## Contents
- `foodscan_ai.ipynb` — full notebook (data import, model, training, prediction)
