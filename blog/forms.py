from django import forms
from .models import Profile


class AdvancedSearchForm(forms.Form):
    title = forms.CharField(required=False, label='Заголовок')
    body = forms.CharField(required=False, widget=forms.Textarea(attrs={'rows': 4}), label='Текст поста')


class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['first_name', 'last_name', 'middle_name', 'avatar']

    def __init__(self, *args, **kwargs):
        super(ProfileForm, self).__init__(*args, **kwargs)
        self.fields['avatar'].widget.attrs['onchange'] = 'previewImage(this);'