from django.urls import path, include
from . import views

urlpatterns = [
    path('classify/', views.classify)
]
