from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('chat.urls')),
    path('admin/', admin.site.urls),
    path('documents/', include('documents.urls')),
    path('accounts/', include('accounts.urls')),
]
