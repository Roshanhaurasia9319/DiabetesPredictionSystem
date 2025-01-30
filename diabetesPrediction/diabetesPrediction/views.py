from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')

def result(request):
    data = pd.read_csv('static\diabetesPrediction\pdf\diabetes.csv')
    X = data.drop('Outcome', axis=1)
    Y = data['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
    model = LogisticRegression()
    model.fit(X_train,Y_train)
    if request.method == 'POST':
        val1 = float(request.POST.get('n1'))
        val2 = float(request.POST.get('n2'))
        val3 = float(request.POST.get('n3'))
        val4 = float(request.POST.get('n4'))
        val5 = float(request.POST.get('n5'))
        val6 = float(request.POST.get('n6'))
        val7 = float(request.POST.get('n7'))
        val8 = float(request.POST.get('n8'))
    
    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    
    result1 = ""
    if pred == 1:
        result1 = "Positive"
    else:
        result1 = "Negative"
        
    return render(request, 'predict.html', {"result2":result1})