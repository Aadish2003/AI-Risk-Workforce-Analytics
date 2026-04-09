# 🤖 AI Workforce Risk Analytics Platform (ML + Streamlit)

## 📌 Project Overview

The **AI Workforce Risk Analytics Platform** is an end-to-end machine learning project that analyzes how Artificial Intelligence is transforming jobs, salaries, and workforce risks between **2020 and 2026**.

This project focuses on building a **custom risk intelligence system** and deploying it through an **interactive Streamlit dashboard**, enabling users to analyze workforce trends and predict job risk levels.

---

## 🎯 Objectives

* Build a custom AI-driven risk scoring system
* Predict job risk levels using machine learning
* Analyze salary transformation due to AI adoption
* Create an interactive dashboard for real-time insights

---

## ⚙️ Custom Risk Engineering (Core Innovation)

Instead of directly using existing labels, a **custom risk score** was engineered:

```
custom_risk_score =
0.35 * skill_gap_index +
0.25 * ai_adoption_level +
0.20 * remote_feasibility_score +
0.10 * wage_volatility_index -
0.15 * years_of_experience
```

### 🔹 Key Logic:

* Higher skill gap → higher risk
* Higher AI adoption → higher disruption
* Higher remote feasibility → more automation exposure
* More experience → lower risk

---

### 🔹 Normalization

The risk score was scaled between **0 and 1** using Min-Max normalization.

---

### 🔹 Risk Classification

The normalized score was converted into categories:

* **0 → Low Risk**
* **1 → Medium Risk**
* **2 → High Risk**

This created a **balanced and realistic target variable** for machine learning.

---

## 🤖 Machine Learning Pipeline

### 🔹 Data Preparation

* Encoded categorical variables using **Label Encoding**
* Applied **StandardScaler** for feature normalization
* Split dataset into **train-test (80-20)**

---

### 🔹 Model Used

* **Random Forest Classifier**

  * n_estimators = 150
  * max_depth = 10

---

### 🔹 Model Performance

* **Accuracy:** 91.49%
* **Cross Validation Scores:**

  * [0.933, 0.917, 0.938, 0.926, 0.898]
* **Mean CV Accuracy:** **92.25%**

---

### 🔹 Interpretation

* The model generalizes well across different data splits
* No major overfitting observed
* Consistent performance across folds

---

## 🧠 Key ML Insight

> The model shows that workforce risk is influenced more by **skills and AI exposure** than just experience.

This highlights the importance of:

* Upskilling
* Adaptability
* AI readiness

---

## 💾 Model Deployment Files

The trained components were saved for deployment:

* `model.pkl` → Random Forest model
* `scaler.pkl` → Feature scaling
* `encoders.pkl` → Label encoding

These are used inside the Streamlit app for real-time predictions.

---

## 📊 Streamlit Application

The project is deployed as an **interactive web application** with the following features:

### 🔹 Dashboard

* Key workforce metrics
* AI adoption and risk overview

### 🔹 Deep Analysis

* Industry-level and role-level insights

### 🔹 Risk Predictor

* Predict job risk using trained ML model

### 🔹 Data Explorer

* Explore dataset dynamically with filters

---

## 🚀 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Random Forest
* Streamlit

---

## 🌐 Deployment

The application is deployed using **Streamlit Cloud**, enabling:

* Real-time predictions
* Interactive filtering
* Scalable analytics

---

## 📌 Conclusion

This project demonstrates how AI is reshaping the workforce by:

* Increasing automation risk
* Changing job stability dynamics
* Emphasizing the importance of skill development

It provides a **data-driven approach to workforce risk analysis** and helps users understand future job trends.

---

## 📬 Author

**Aadish Kotadia**
MBA (Business Analytics)

📧 [kotadiaaadish1234@gmail.com](mailto:kotadiaaadish1234@gmail.com)
🔗 LinkedIn: www.linkedin.com/in/aadish-kotadia-9a4957241
💻 GitHub: https://github.com/Aadish2003

---

## ⭐ If you like this project, give it a star!
