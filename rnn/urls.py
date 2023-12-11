from django.urls import path
from .views import ReviewListView, model_view, SearchView

urlpatterns = [
    path('dataset/', ReviewListView.as_view(), name='dataset'),
    path('model/', model_view, name='model'),
    path('search/', SearchView.search_view, name='search')
]