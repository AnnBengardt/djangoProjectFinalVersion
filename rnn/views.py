from django.shortcuts import render, redirect
from django.views.generic import ListView, TemplateView
from .models import Review
from .model import predict


class ReviewListView(ListView):
    model = Review
    context_object_name = 'page_obj'
    template_name = 'dataset.html'


def model_view(request):
    if request.method == 'POST':
        row = request.POST.get('row')
        obj = Review.objects.filter(row=row)
        text = obj.values("text")[0]["text"]
        y_pred = predict(text)
        obj.update(prediction=y_pred)
        return render(request, 'model.html', {'prediction': y_pred, "text": text})
    else:
        return render(request, 'model.html')


class SearchView(TemplateView):
    def search_view(request):
        query = request.GET.get('q')

        if not query:
            return redirect('home')  # переход на главную, если запрос пустой

        results = Review.objects.filter(text__icontains=query)

        context = {'results': results, 'query': query, "item": "review"}
        return render(request, 'search_results.html', context)

