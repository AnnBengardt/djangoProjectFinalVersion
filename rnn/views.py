from django.shortcuts import render, redirect
from django.views.generic import ListView, TemplateView
from .models import Review
from .model import predict, train


class ReviewListView(ListView):
    model = Review
    context_object_name = 'page_obj'
    template_name = 'dataset.html'

class ModelView(TemplateView):
    def model_view(request):
        if request.method == 'POST':
            row = request.POST.get('row')
            obj = Review.objects.filter(row=row)
            text = obj.values("text")[0]["text"]
            label = obj.values("label")[0]["label"]
            text_label = "отрицательный" if label == 0 else "положительный"
            y_pred = predict(text)
            obj.update(prediction=y_pred)
            return render(request, 'model.html', {'prediction': y_pred, "text": text, "label": text_label})
        else:
            return render(request, 'model.html')


class TrainView(TemplateView):
    def train_view(request):
        if request.method == 'POST':
            epochs = request.POST.get('epochs')
            embed = request.POST.get('embed')
            hidden = request.POST.get('hidden')
            layers = request.POST.get('layers')
            data = [review["text"] for review in Review.objects.all().values("text")]
            labels = [review["label"] for review in Review.objects.all().values("label")]
            res = train(int(epochs), data, labels, int(embed), int(hidden), int(layers))
            history = []
            for epoch in range(res["epochs"]):
                history.append({"epoch":epoch+1,"train_loss":res["train_loss"][epoch], "train_acc":res["train_acc"][epoch]})
            return render(request, 'train.html', {'history': history})
        else:
            return render(request, 'train.html')


class SearchView(TemplateView):
    def search_view(request):
        query = request.GET.get('q')

        if not query:
            return redirect('home')  # переход на главную, если запрос пустой

        results = Review.objects.filter(text__icontains=query)

        context = {'results': results, 'query': query, "item": "review"}
        return render(request, 'search_results.html', context)

