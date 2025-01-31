# Wine Quality Prediction: Neural Network Models

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)

## Description

This repository contains three Python scripts demonstrating the application of machine learning techniques to predict the quality of wine using the "winequality_data.csv" dataset. The code includes implementations for classification tasks using neural networks in PyTorch, with varying configurations and enhancements like early stopping and SMOTE. It also include two notebooks: one with data analysis and the other with other models.

The scripts include:

- `baseline_model.py`: A basic neural network model for wine quality prediction.
- `binary_classification.py`: A binary classification model that predicts wine quality as either 'low' or 'high'.
- `multiclass_classification.py`: A multi-class classification model for predicting 'low', 'average', or 'high' wine quality.

## Installation

1. Install dependencies:

Python 3.7 or higher
PyTorch
Pandas
Scikit-learn
Matplotlib
Seaborn
Imbalanced-learn (for SMOTE)
tqdm

## Usage

### Data

The code requires the "winequality_data.csv" dataset. You can download it from the UCI Machine Learning Repository.

1. baseline_model.py
This script demonstrates a simple neural network model for wine quality prediction, using multiple classes (low, average, and high) as the target variable. The model is trained for 200 epochs with Adam optimizer, CrossEntropyLoss, and tracks accuracy and loss.

2. binary_classification.py
This script focuses on binary classification, where the quality of wine is classified into two categories: low and high. The dataset is preprocessed, with an EarlyStopping class implemented to prevent overfitting and optimize training. The training continues for a maximum of 1000 epochs (or until early stopping is triggered).

3. multiclass_classification.py
In this script, the model predicts wine quality as low, average, or high, similar to baseline_model.py but with the added enhancement of using SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.

## Contributors

- [BeatrizJover](https://github.com/BeatrizJover)