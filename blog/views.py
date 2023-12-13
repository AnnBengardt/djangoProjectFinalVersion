from bootstrap_modal_forms.generic import BSModalCreateView, BSModalLoginView
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render, get_object_or_404
from django.urls import reverse_lazy
from django.views.generic import ListView, TemplateView
from .models import Post
from django.core.paginator import Paginator
from .forms import AdvancedSearchForm, ProfileForm, CustomAuthenticationForm, CustomUserCreationForm
from .models import Profile

class BlogListView(ListView):
    model = Post
    paginate_by = 3
    template_name = 'home.html'


class AboutPageView(TemplateView):
    template_name = 'about.html'


class ImputPageView(TemplateView):
    template_name = 'imput.html'

class SearchView(TemplateView):
    def search_view(request):
        query = request.GET.get('q')

        if not query:
            return redirect('home')  # переход на главную, если запрос пустой

        results = Post.objects.filter(title__icontains=query) | Post.objects.filter(body__icontains=query)

        context = {'results': results, 'query': query, 'item': "post"}
        return render(request, 'search_results.html', context)


class AdvancedSearchView(TemplateView):
    def advanced_search_view(request):
        form = AdvancedSearchForm(request.GET)
        results = []

        if form.is_valid():
            title_query = form.cleaned_data.get('title', '')
            body_query = form.cleaned_data.get('body', '')

            results = Post.objects.filter(title__icontains=title_query, body__icontains=body_query)

        context = {'form': form, 'results': results}
        return render(request, 'advanced_search.html', context)


class ProfileView(TemplateView):

    @login_required
    def profile_view(request):
        try:
            profile = request.user.profile
        except Profile.DoesNotExist:
            profile = None

        if request.method == 'POST':
            form = ProfileForm(request.POST, request.FILES, instance=profile)
            if form.is_valid():
                profile = form.save(commit=False)
                profile.user = request.user
                profile.save()
        else:
            form = ProfileForm(instance=profile)

        return render(request, 'profile.html', {'form': form, 'profile': profile})


class SignUpView(BSModalCreateView):
    form_class = CustomUserCreationForm
    template_name = 'signup.html'
    success_message = 'Success: Sign up succeeded. You can now Log in.'
    success_url = reverse_lazy('home')




class CustomLoginView(BSModalLoginView):
    authentication_form = CustomAuthenticationForm
    template_name = 'registration/login.html'
    success_message = 'Success: You were successfully logged in.'
    extra_context = dict(success_url=reverse_lazy('home'))


class PostView(TemplateView):
    def get_post_by_title(request, title):
        post = get_object_or_404(Post, title=title)
        return render(request, 'post.html', {'post': post})