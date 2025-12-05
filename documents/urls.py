from django.contrib import admin
from django.urls import path, include
from .views import dashboard, delete_document

app_name = 'documents'
urlpatterns = [
    path('', dashboard, name='dashboard'),
    path('delete/<int:document_id>/', delete_document, name='delete_document'),
]
