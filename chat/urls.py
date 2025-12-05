from django.contrib import admin
from django.urls import path, include
from .views import index

app_name = "chat"

urlpatterns = [
    path('', index, name='index'),
]
