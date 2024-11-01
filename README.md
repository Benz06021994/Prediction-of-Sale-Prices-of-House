# Real Estate Price Prediction

Source : [Please find the Dataset here](https://data.world/dataquest/nyc-property-sales-data)

## Overview

The House Price Prediction Model project aims to harness the power of machine learning to accurately forecast real estate prices. By analyzing a comprehensive dataset that includes various features such as location, size, number of rooms, and other relevant factors, this project seeks to develop a predictive model that can provide valuable insights for potential homebuyers, real estate agents, and investors.

The project will involve several key stages, including data collection, preprocessing, exploratory data analysis (EDA), model selection, training, evaluation, and deployment. Various machine learning algorithms will be explored and compared to identify the most effective approach for predicting house prices. The final model will be evaluated based on performance metrics such as RMSE, MAE, and R² to ensure high accuracy and reliability.

In addition to building the predictive model, the project will also focus on the practical application of the model and strategies for deployment. The ultimate goal is to create a user-friendly tool that can assist stakeholders in making informed decisions in the real estate market.

## Objective

The primary objective of this project is to develop a robust and accurate machine learning model capable of predicting house prices based on various features such as location, size and other relevant factors. This model aims to provide valuable insights for potential homebuyers, real estate agents, and investors by leveraging data-driven techniques to forecast property values. Additionally, the project seeks to explore and compare different machine learning algorithms to identify the most effective approach for house price prediction, ensuring high accuracy and reliability in the predictions.

## Table of Contents

- [Introduction](#introduction)
- [Data Description](#data-description)
- [Data Preprocessing](#data-preprocessing)
- [Feature Selection](#feature-selection)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Deployment](#model-deployment)
- [Pipeline Implementation](#pipeline-implementation)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Clone](#clone)
- [Contributing](#contributing)

## Introduction

The goal of this project is to build a machine learning model that can accurately predict real estate prices based on various features. This can be useful for real estate agents, buyers, and investors to make informed decisions.

## Data Description

Here's a detailed description of the features used in your real estate price prediction project:

1. **Borough**:
   - **Description**: The borough where the property is located.
   - **Type**: Categorical

2. **Neighborhood**:
   - **Description**: The specific neighborhood within the borough.
   - **Type**: Categorical

3. **Building Class Category**:
   - **Description**: The category of the building class (e.g., residential, commercial).
   - **Type**: Categorical

4. **Tax Class at Present**:
   - **Description**: The current tax classification of the property.
   - **Type**: Categorical

5. **Block**:
   - **Description**: The block number where the property is located.
   - **Type**: Numerical

6. **Lot**:
   - **Description**: The lot number within the block.
   - **Type**: Numerical

7. **Building Class at Present**:
   - **Description**: The current building class designation.
   - **Type**: Categorical

8. **Address**:
   - **Description**: The address of the property.
   - **Type**: Categorical

9. **Residential Units**:
   - **Description**: The number of residential units in the property.
   - **Type**: Numerical

10. **Commercial Units**:
    - **Description**: The number of commercial units in the property.
    - **Type**: Numerical

11. **Total Units**:
    - **Description**: The total number of units (residential + commercial) in the property.
    - **Type**: Numerical

12. **Year Built**:
    - **Description**: The year the property was built.
    - **Type**: Numerical

13. **Gross Square Feet**:
    - **Description**: The total gross square footage of the property.
    - **Type**: Numerical

14. **Land Square Feet**:
    - **Description**: The total land square footage of the property.
    - **Type**: Numerical

15. **Sale Price**:
    - **Description**: The sale price of the property.
    - **Type**: Numerical (Target Variable)

## Data Preprocessing

- **Label Encoding**: Categorical features were encoded using label encoding.
- **Feature Scaling**: StandardScaler was used to standardize the features.
- **Data Splitting**: The dataset was split into training and testing sets.

## Feature Selection

- **RandomForestRegressor**: Used to identify important features.
- **Selected Features**: 'gross_square_feet', 'block', 'borough', among others.

## Model Training and Evaluation

Multiple regression models were trained and evaluated:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**

### Evaluation Metrics

- **R2 Score**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

The Random Forest Regressor emerged as the best performing model based on these metrics.

## Hyperparameter Tuning

- **GridSearchCV**: Used for hyperparameter tuning of the Random Forest model.
- **Best Parameters**: Identified and applied to improve model performance.

- After Hyperparameter Tuning, Random Forest Regressor found to be most suitable model.

## Model Deployment

- **Model Saving**: The best performing model was saved.

## Pipeline Implementation

- **Predictions**: Made predictions on unseen data using pipeline and displayed the predictions

## Conclusion

The project successfully demonstrates the process of building a machine learning model for real estate price prediction. The XGBoost Regressor, after hyperparameter tuning, provided the best results.

## Future Work

To further improve the model, consider the following:
- **Feature Engineering**: Explore additional features and create interaction features.
- **Advanced Models**: Experiment with models like LightGBM, CatBoost, or neural networks.
- **Data Augmentation**: Increase dataset size and handle imbalanced data.
- **Model Interpretability**: Implement SHAP values and LIME for better interpretability.
- **Cross-Validation**: Use k-fold cross-validation for robustness.
- **Regular Updates**: Update the model with new data regularly.

## Clone

 Clone the repository:
  
   git clone https://github.com/yourusername/real-estate-price-prediction.git
   

## Contributing

Contributions are welcome! 

