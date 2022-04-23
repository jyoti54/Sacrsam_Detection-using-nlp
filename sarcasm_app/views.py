from django.shortcuts import render
from joblib import load
model= load('./savedModels/model.joblib')
# Create your views here.

def predict_sarcasm(request):
    return render(request, 'main.html')

def formInfo(request):
    t1=request.GET['text1']


    y_pred=model.predict([[t1]])
    print(y_pred)

    if y_pred == 0:
        print("Sarcastic")
    else:
        print("Not Sarcastic")

    return render(request, 'result.html', {'result' : y_pred})




