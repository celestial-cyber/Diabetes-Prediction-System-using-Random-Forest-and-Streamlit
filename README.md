# Diabetes-Prediction-System-using-Random-Forest-and-Streamlit
This project is a machine learning-based web app that predicts the likelihood of diabetes in individuals based on key health indicators. It uses a Random Forest Classifier trained on the Pima Indians Diabetes dataset and is deployed using an interactive Streamlit interface.

# Diabetes Prediction System using Random Forest and Streamlit

This project is an end-to-end machine learning application designed to predict the likelihood of diabetes in individuals based on key health indicators. It is built using a Random Forest Classifier and deployed through an interactive Streamlit web interface, enabling real-time predictions from user inputs.

## 🔍 Overview

The model is trained on the Pima Indians Diabetes dataset, a widely-used dataset in medical research. After data preprocessing and outlier handling using IQR-based Winsorization, a Random Forest classifier is trained to distinguish diabetic and non-diabetic individuals. The application allows users to enter patient details and view prediction results along with model accuracy and a confusion matrix for evaluation.

## 🧠 Technologies and Rationale

- **Python (Pandas, NumPy):** For data manipulation and numerical operations.
- **Seaborn, Matplotlib:** For exploratory data analysis and plotting.
- **Feature-engine (Winsorizer):** To cap outliers using IQR method, enhancing model performance by handling extreme values in features like Insulin, BMI, Blood Pressure, etc.
- **Scikit-learn:**
  - `RandomForestClassifier`: Chosen for its robustness and ability to handle feature importance and non-linear data.
  - `train_test_split`: To divide data into training and testing subsets for evaluation.
  - `accuracy_score`, `confusion_matrix`: To assess model performance.
- **Streamlit:** To build a simple, browser-based interface where users can input values and view predictions instantly.

## ⚙️ Features

- Cleans the dataset using IQR-based Winsorization  
- Trains a Random Forest model with optimized hyperparameters  
- Takes 8 user inputs, including Glucose, Insulin, BMI, etc.  
- Calculatesthe  probability of being diabetic  
- Real-time predictions via Streamlit web app  
- Displays accuracy scores and a confusion matrix in the interface

## 📥 User Inputs

- Age  
- Number of Pregnancies  
- Glucose Level  
- Blood Pressure  
- Skin Thickness  
- Insulin Level  
- BMI (Body Mass Index)  
- Diabetes Pedigree Function

## 🛠️ How to Run This Project

1. **Clone or download the repository** and save the code as `diabetes_app.py`.
2. **Install required libraries**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn streamlit feature_engine

   Open Command Prompt or Terminal and navigate to the project folder:

3. Open Command Prompt or Terminal and navigate to the project folder:
   cd path\to\your\project
4. Run the app using Streamlit:
   python -m streamlit run diabetes_app.py

