from django.urls import path
from .views import ReviewListView, ModelView, SearchView, TrainView

urlpatterns = [
    path('dataset/', ReviewListView.as_view(), name='dataset'),
    path('model/', ModelView.model_view, name='model'),
    path('rev_search/', SearchView.search_view, name='rev_search'),
    path('train/', TrainView.train_view, name='train')
]