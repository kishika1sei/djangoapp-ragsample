from django.contrib import admin
from django.urls import path, include
from .views import login_view as login, logout_view as logout

app_name = "accounts"

urlpatterns = [
    path("login/", login, name="login"),
    path("logout/", logout, name="logout"),
]
