from django.urls import path
from .views import index,reset_view,submit_rating

app_name = "chat"

urlpatterns = [
    path('', index, name='index'),
    path('reset/', reset_view, name="reset"),
    path('submit_rating/', submit_rating, name="submit_rating"),
]
