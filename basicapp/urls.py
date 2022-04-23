from django.urls import URLPattern, path
from . import views

urlpatterns = [
    path('', views.Welcome,name='Welcome'),
    path('user', views.User, name='user'),
    ]
