from django.views.generic import ListView, TemplateView
from .models import Post
from django.core.paginator import Paginator

class BlogListView(ListView):
    model = Post
    paginate_by = 2
    template_name = 'home.html'


class AboutPageView(TemplateView):
    template_name = 'about.html'


class ImputPageView(TemplateView):
    template_name = 'imput.html'


