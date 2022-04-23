from django.urls import path
from . import views

urlpatterns = [
path('', views.predict_sarcasm, name='predict_sarcasm'),
path('result', views.formInfo, name='result')

]