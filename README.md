# 💳 Credit Scoring & Creditworthiness Prediction

## 📌 Project Overview

This project is a Machine Learning based Credit Scoring and Creditworthiness Prediction system developed as part of my CodeAlpha Internship.

The system predicts whether an applicant is likely to be **Creditworthy** or **Higher Risk** based on financial and credit-related information.

An interactive Streamlit web application was developed to allow users to enter applicant details and receive a prediction along with probability and risk assessment.

## 🎯 Objectives

- Predict an applicant's creditworthiness using Machine Learning.
- Perform feature engineering on financial data.
- Compare different classification algorithms.
- Evaluate the models using multiple performance metrics.
- Build an interactive Streamlit application for real-time predictions.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- Streamlit

## 🤖 Machine Learning Models

The following classification algorithms were evaluated:

- Logistic Regression
- Random Forest Classifier

Logistic Regression was selected for the final application because it achieved better performance on the test dataset.

## 📊 Model Performance

| Metric | Logistic Regression |
|---|---:|
| Accuracy | 70.25% |
| Precision | 72.16% |
| Recall | 84.68% |
| F1 Score | 77.92% |
| ROC-AUC | 76.20% |

## 📋 Features Used

The model uses the following features:

- Annual Income
- Total Debt
- Payment History Score
- Applicant Age
- Number of Open Credit Accounts
- Debt-to-Income Ratio
- Score-to-Debt Ratio

## 🌐 Streamlit Application

The Streamlit application provides:

- Applicant financial information input
- Creditworthiness prediction
- Creditworthy probability
- Higher-risk probability
- Financial risk assessment
- Interactive prediction gauge

## 🚀 How to Run Locally

Clone this repository:

```bash
git clone https://github.com/YOUR-USERNAME/CodeAlpha_CreditScoringModel.git
