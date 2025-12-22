from django.contrib import admin
from django.urls import path, include
from .views import dashboard, delete_document,reindex_all

app_name = 'documents'
urlpatterns = [
    path('', dashboard, name='dashboard'),
    path('delete/<int:document_id>/', delete_document, name='delete_document'),
    path("reindex-all/", reindex_all, name="reindex_all"),
]
