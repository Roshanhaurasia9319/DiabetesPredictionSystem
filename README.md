# Diabetes Prediction System

## Overview
The **Diabetes Prediction System** is a web-based application built using Django and Machine Learning to predict whether a person has diabetes based on medical attributes. It utilizes **Logistic Regression** for classification and is trained on a diabetes dataset.

## Features
- User-friendly web interface
- Predicts whether a person has diabetes based on input values
- Uses **Logistic Regression** for prediction
- Displays prediction result as "Positive" or "Negative"
- Data processing using **pandas**, **matplotlib**, and **seaborn**

## Technologies Used
- **Django** (Web Framework)
- **Python** (Programming Language)
- **pandas** (Data Manipulation)
- **matplotlib & seaborn** (Data Visualization)
- **scikit-learn** (Machine Learning)

## Installation
### Prerequisites
Make sure you have Python and pip installed. You can install the required dependencies using:
```bash
pip install django pandas matplotlib seaborn scikit-learn
```

### Clone the Repository
```bash
git clone https://github.com/your-username/diabetes-prediction-system.git
cd diabetes-prediction-system
```

### Run the Django Server
```bash
python manage.py runserver
```
Then, open `http://127.0.0.1:8000/` in your web browser.

## Usage
1. Open the application in a web browser.
2. Enter the required medical details (e.g., Glucose, Blood Pressure, BMI, etc.).
3. Click **Predict** to get the result.
4. The system will display whether the person is **Diabetes Positive** or **Diabetes Negative**.

## Project Structure
```
Diabetes-Prediction-System/
│── diabetes_prediction/
│   ├── static/
│   │   ├── diabetesPrediction/pdf/diabetes.csv  # Dataset
│   ├── templates/
│   │   ├── home.html
│   │   ├── predict.html
│   ├── views.py  # Django views
│── manage.py  # Django management script
│── README.md  # Project Documentation
```

## Machine Learning Model
- **Dataset**: The model is trained on a diabetes dataset (`diabetes.csv`).
- **Algorithm**: Logistic Regression
- **Evaluation**: Uses `train_test_split` to split data (70% training, 30% testing)
- **Prediction**: Based on user input, the model predicts diabetes outcome (`Positive` or `Negative`).

## Future Improvements
- Implement additional machine learning models (Random Forest, SVM, etc.)
- Improve UI with Bootstrap for a better user experience
- Deploy the application on a cloud platform


