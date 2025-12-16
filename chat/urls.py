from django.contrib import admin
from django.urls import path, include
from .views import index,reset_view

app_name = "chat"

urlpatterns = [
    path('', index, name='index'),
    path('reset/', reset_view, name="reset"),
]
