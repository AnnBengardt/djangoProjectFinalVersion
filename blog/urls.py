from django.urls import path
from blog import views
from .views import BlogListView, AboutPageView, ImputPageView, SearchView, AdvancedSearchView, ProfileView

urlpatterns = [
    path('', BlogListView.as_view(), name='home'),
    path('about/', AboutPageView.as_view(), name='about'),
    path('imput/', ImputPageView.as_view(), name='imput'),
    path('search/', SearchView.search_view, name='search'),  # Обычный поиск
    path('advanced_search/', AdvancedSearchView.advanced_search_view, name='advanced_search'),
    path('profile/', ProfileView.profile_view, name='profile')
]
