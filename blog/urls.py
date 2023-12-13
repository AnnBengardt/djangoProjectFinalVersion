from django.urls import path
from blog import views
from .views import BlogListView, AboutPageView, ImputPageView, SearchView, AdvancedSearchView, ProfileView, SignUpView, CustomLoginView, PostView

urlpatterns = [
    path('', BlogListView.as_view(), name='home'),
    path('about/', AboutPageView.as_view(), name='about'),
    path('imput/', ImputPageView.as_view(), name='imput'),
    path('search/', SearchView.search_view, name='search'),
    path('advanced_search/', AdvancedSearchView.advanced_search_view, name='advanced_search'),
    path('profile/', ProfileView.profile_view, name='profile'),
    path('signup/', SignUpView.as_view(), name='signup'),
    path("login/", CustomLoginView.as_view(), name="login"),
    path('<str:title>/', PostView.get_post_by_title, name="post")
]
