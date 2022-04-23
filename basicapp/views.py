from django.shortcuts import render
from django.http import HttpResponse


def Welcome(request):
    return render(request, 'index.html')

def User(request):
    usertext=request.GET['text']
    # print(usertext)
    return render(request, 'user.html', {'name':usertext})


