# Mobile Price Prediction Project

## Overview

This project aims to predict mobile phone prices based on their specifications using machine learning. It explores three different models (Random Forest, Neural Networks, and Gradient Boosting) with two hyperparameter tuning strategies (random and grid search). The dataset used is from Kaggle and includes features like brand, ratings, RAM, storage, and camera specifications.

## Key Steps

1. **Data Preprocessing:**
   - **Handling Missing Values:** Filling missing values with appropriate strategies.
   - **Outlier Removal:** Removing extreme outliers to avoid skewing the analysis.
   - **One-Hot Encoding:** Converting categorical variables (brand) into numerical format for model compatibility.
   - **Feature Selection:** Selecting relevant features based on their correlation with the target variable (price).

2. **Feature Scaling:**
   - **Min-Max Scaling:** Normalizing numerical features to a 0-1 range to ensure all features contribute equally to the model.

3. **Model Selection and Hyperparameter Tuning:**
   - **Models:** Random Forest, Neural Networks (MLP), and Gradient Boosting.
   - **Tuning:** RandomizedSearchCV and GridSearchCV used to find optimal hyperparameters for each model.

4. **Model Training and Evaluation:**
   - **Training:** Models are trained on a training set using the best hyperparameters found during tuning.
   - **Evaluation:** Models are evaluated on a validation set using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) metrics. The best model is selected based on validation performance.

## Results and Insights

- **Best Model:** Random Forest consistently outperformed other models, achieving the highest R-squared values on both training and validation sets. This suggests that Random Forests are well-suited for predicting mobile phone prices based on specifications.
- **Feature Importance:**  The model identified `Ratings`, `Brand_Apple`, `ROM`, and `Brand_Samsung` as the most important predictors of mobile phone prices.

## Future Work

* **Advanced Feature Engineering:** Explore feature interactions and create new features to potentially improve predictive power.
* **Additional Models:** Experiment with other regression algorithms like Support Vector Regression or ensemble methods like stacking.
* **More Data:** Collecting a larger and more diverse dataset could further improve model generalization and performance.


## Important Links
* **Dataset** - https://www.kaggle.com/datasets/jsonali2003/mobile-price-prediction-dataset
* **Model Training Code** - https://colab.research.google.com/drive/1x2uRHIiQaupq1-65wWAtJjGJFjkqPeaM?usp=sharing
* **Website Application** - https://led-mobile-price-prediction.streamlit.app/
