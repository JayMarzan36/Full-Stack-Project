from django.urls import path
from . import views

#TODO add urls for other views?
urlpatterns = [
    path('', view=views.index, name="index"),
]