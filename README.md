# End to End Ml Project

# 🎯 Student Performance Indicator

A complete machine learning project designed to **predict students' test performance** using various demographic and preparatory variables. This project walks through the end-to-end lifecycle of a typical ML workflow — from data understanding to deployment — and showcases model comparisons to choose the best regressor.

---

## 🔍 Problem Statement

The goal of this project is to understand how a student's test performance is influenced by features such as:
- Gender
- Ethnicity
- Parental level of education
- Lunch type
- Test preparation course

The final output is the predicted average test score.

---

## 📊 Dataset

- **Source**: [Kaggle Dataset – Student Performance](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)
- **Shape**: 1000 rows × 8 columns

---

## 🔁 ML Project Pipeline

1. Understanding the Problem Statement  
2. Data Collection  
3. Data Validation & Cleaning  
4. Exploratory Data Analysis (EDA)  
5. Data Preprocessing  
6. Model Building & Evaluation  
7. Model Selection  
8. Deployment (AWS & Azure with CI/CD)  

---

## 🧠 Models Used & Performance Summary

A variety of regression models were evaluated using RMSE, MAE, and R² on both training and test sets.

### ✅ Best Performing Models (Test R² Score)

| Model                    | Test R² Score | Test RMSE | Test MAE |
|-------------------------|---------------|-----------|----------|
| **Linear Regression**    | 0.8804        | 5.3940    | 4.2148   |
| **Ridge Regression**     | **0.8806**    | 5.3904    | 4.2111   |
| CatBoost Regressor      | 0.8516        | 6.0086    | 4.6125   |
| AdaBoost Regressor      | 0.8524        | 5.9923    | 4.6832   |
| Random Forest Regressor | 0.8482        | 6.0779    | 4.7210   |

Other models like XGBoost and Decision Tree showed high variance and possible overfitting.

---

## 🚀 Deployment

The project is containerized and deployed to cloud platforms:
- **Amazon Web Services (AWS)**
- **Microsoft Azure**

CI/CD pipelines were implemented using:
- **GitHub Actions**
- **Docker**
- **Azure Pipelines / AWS CodePipeline**

This ensures reliable and automated deployment with each new update.

---

## 📦 Technologies Used

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Regressors: Linear, Ridge, Lasso, KNN, Decision Tree, Random Forest, XGBoost, CatBoost, AdaBoost
- Docker, GitHub Actions
- AWS EC2 / S3, Azure Web Apps
- CI/CD Tools for automated deployment

---

