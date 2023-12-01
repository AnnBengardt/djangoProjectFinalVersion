from django.urls import path
from blog import views
from .views import BlogListView, AboutPageView, ImputPageView

urlpatterns = [
    path('', BlogListView.as_view(), name='home'),
    path('about/', AboutPageView.as_view(), name='about'),
    path('imput/', ImputPageView.as_view(), name='imput')
]
