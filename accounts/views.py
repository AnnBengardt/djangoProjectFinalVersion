from django.shortcuts import render

# accounts/views.py
from django.utils.decorators import method_decorator


from django.urls import reverse_lazy
from bootstrap_modal_forms.generic import BSModalCreateView, BSModalLoginView
from django.views.decorators.csrf import csrf_exempt

from .forms import CustomUserCreationForm, CustomAuthenticationForm

@method_decorator(csrf_exempt, name='dispatch')
class SignUpView(BSModalCreateView):
    form_class = CustomUserCreationForm
    template_name = 'signup.html'
    success_message = 'Success: Sign up succeeded. You can now Log in.'
    success_url = reverse_lazy('home')

@method_decorator(csrf_exempt, name='dispatch')
class CustomLoginView(BSModalLoginView):
    authentication_form = CustomAuthenticationForm
    template_name = 'login.html'
    success_message = 'Success: You were successfully logged in.'
    extra_context = dict(success_url=reverse_lazy('home'))
