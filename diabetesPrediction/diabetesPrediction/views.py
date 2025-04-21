from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Machine Learning imports
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# Other imports
from typing import Tuple  # Optional, for type hints

def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')

# def result(request):
#     data = pd.read_csv('static\diabetesPrediction\pdf\diabetes.csv')
#     X = data.drop('Outcome', axis=1)
#     Y = data['Outcome']
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
#     model = LogisticRegression()
#     model.fit(X_train,Y_train)
#     if request.method == 'POST':
#         val1 = float(request.POST.get('n1'))
#         val2 = float(request.POST.get('n2'))
#         val3 = float(request.POST.get('n3'))
#         val4 = float(request.POST.get('n4'))
#         val5 = float(request.POST.get('n5'))
#         val6 = float(request.POST.get('n6'))
#         val7 = float(request.POST.get('n7'))
#         val8 = float(request.POST.get('n8'))
    
#     pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    
#     result1 = ""
#     if pred == 1:
#         result1 = "Positive"
#     else:
#         result1 = "Negative"
        
#     return render(request, 'predict.html', {"result2":result1})

# Knowledge Base for Recommendations

data = pd.read_csv('static\diabetesPrediction\pdf\diabetes.csv')
# Preprocessing
X = data.drop(columns=['Outcome'])
Y = data['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.30, random_state=42)
# Train the model
# model = LogisticRegression(max_iter=200)
# model.fit(X_train, Y_train)


model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', random_state=42)
model.fit(X_train, Y_train)
# mlp_pred = mlp.predict(X_test)

def knowledge_base():
    return {
        "exercise": {
            "young": "Brisk walking, jogging, strength training",
            "middle-aged": "Brisk walking, yoga, moderate cardio",
            "elderly": "Light walking, stretching, tai chi"
        },
        "diet": {
            "underweight": "Balanced diet with sufficient calories, healthy fats",
            "normal": "Balanced diet, low sugar, high fiber",
            "overweight": "Low carb, high fiber, moderate protein",
            "obese": "Strict low glycemic diet, portion control"
        },
        "medication": {
            "Normal": "No medication needed",
            "Mild Diabetes": "Lifestyle changes, possible metformin",
            "Moderate Diabetes": "Metformin, insulin if required",
            "Severe Diabetes": "Insulin therapy required, strict monitoring"
        }
    }

# Decision Tree for Blood Pressure Analysis
def analyze_blood_pressure(bp, glucose, age):
    """
    Decision tree for blood pressure analysis.
    
    Parameters:
    - bp (int): Blood Pressure level
    - glucose (int): Glucose level
    - age (int): Age of the person
    
    Returns:
    - Tuple: (BP case, risk level, recommendation)
    """

    # Tree Structure
    if age < 40:
        if bp >= 140:
            return ("Hypertension (High BP)", "High Risk for Heart & Kidney Disease", "Reduce salt intake, exercise regularly.")
        elif bp <= 90:
            return ("Hypotension (Low BP)", "Risk of Dizziness & Poor Circulation", "Increase fluid and salt intake.")
        elif 90 < bp < 140 and glucose > 200:
            return ("Fluctuating BP due to Diabetes", "Risk of Stroke, Poor Diabetes Control", "Monitor BP & glucose regularly.")
        else:
            return ("Normal Blood Pressure", "No Immediate Risk", "Maintain a healthy lifestyle.")

    elif 40 <= age < 60:
        if bp >= 140:
            return ("Hypertension (Moderate)", "Risk for Heart Disease", "Exercise daily, reduce salt intake, monitor BP.")
        elif bp <= 90:
            return ("Low BP", "Dizziness Risk", "Drink water, check medication effects.")
        else:
            return ("Stable BP", "No Immediate Risk", "Maintain healthy habits.")

    else:
        if bp >= 140:
            return ("Hypertension (Severe)", "High Risk for Stroke", "Monitor BP daily, consult a doctor immediately.")
        elif bp <= 90:
            return ("Hypotension", "Circulatory Risk", "Increase fluid intake, avoid standing up too fast.")
        else:
            return ("Normal BP", "Low Risk", "Continue regular check-ups and a healthy lifestyle.")

# Decision Tree for Diabetes Stage Inference
def infer_diabetes_stage(glucose, age, bmi):
    """
    Decision tree for diabetes stage and recommendations.
    
    Parameters:
    - glucose (int): Glucose level
    - age (int): Age of the person
    - bmi (float): Body Mass Index (BMI)
    
    Returns:
    - Tuple: (Diabetes stage, recommended exercise, recommended diet, medication)
    """

    # Knowledge Base
    kb = knowledge_base()

    # Diabetes Stage Tree
    if glucose < 140:
        stage = "Normal"
    elif 140 <= glucose < 180:
        stage = "Mild Diabetes"
    elif 180 <= glucose < 250:
        stage = "Moderate Diabetes"
    else:
        stage = "Severe Diabetes"

    # Exercise Recommendation Tree
    if age < 40:
        exercise = kb["exercise"]["young"]
    elif 40 <= age < 60:
        exercise = kb["exercise"]["middle-aged"]
    else:
        exercise = kb["exercise"]["elderly"]

    # Diet Recommendation Tree
    if bmi < 18.5:
        diet = kb["diet"]["underweight"]
    elif 18.5 <= bmi < 25:
        diet = kb["diet"]["normal"]
    elif 25 <= bmi < 30:
        diet = kb["diet"]["overweight"]
    else:
        diet = kb["diet"]["obese"]

    # Medication Recommendation Tree
    medication = kb["medication"][stage]

    return stage, exercise, diet, medication



def result(request):
    """Handles user input, makes ML prediction, and provides detailed expert recommendations."""
    if request.method == 'POST':
        try:
            # Get user inputs
            val1 = float(request.POST.get('n1'))  # Pregnancies
            val2 = float(request.POST.get('n2'))  # Glucose
            val3 = float(request.POST.get('n3'))  # BloodPressure
            val4 = float(request.POST.get('n4'))  # SkinThickness
            val5 = float(request.POST.get('n5'))  # Insulin
            val6 = float(request.POST.get('n6'))  # BMI
            val7 = float(request.POST.get('n7'))  # DiabetesPedigreeFunction
            val8 = float(request.POST.get('n8'))  # Age

            # Check for history of gestational diabetes (optional)
            gdm_history = request.POST.get('gdm_history', 'no').lower() == 'yes'

            # Prepare input for ML model
            input_data = scaler.transform([[val1, val2, val3, val4, val5, val6, val7, val8]])
            prediction = model.predict(input_data)[0]      

            # Analyze Blood Pressure
            bp_case, bp_risk, bp_recommendation = analyze_blood_pressure(val3, val2, val8)

            # Infer diabetes stage & recommendations
            diabetes_stage, exercise, diet, medication = infer_diabetes_stage(val2, val8, val6)

            # Set result message
            result1 = "Positive" if prediction == 1 else "Negative"
            result_message = {
                "prediction": result1,
                "diabetes_stage": diabetes_stage if prediction == 1 else "Not Applicable",
                "exercise": exercise if prediction == 1 else "No changes needed",
                "diet": diet if prediction == 1 else "Maintain a balanced diet",
                "medication": medication if prediction == 1 else "No medication needed",
                "blood_pressure_case": bp_case,
                "blood_pressure_risk": bp_risk,
                "blood_pressure_recommendation": bp_recommendation,
                "warning": "Gestational diabetes history detected. Monitor closely." if gdm_history and prediction == 1 else "No pregnancy-related risk detected."
            }

            return render(request, 'result.html', {"result": result_message})

        except Exception as e:
            return render(request, 'result.html', {"error": f"Error: {str(e)}"})

    return render(request, 'result.html')
