from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models

class Review(models.Model):
    row = models.IntegerField(primary_key=True)
    text = models.TextField()
    label = models.IntegerField(validators=[
        MinValueValidator(0),
        MaxValueValidator(1)])
    prediction = models.IntegerField(blank=True, validators=[
        MinValueValidator(0),
        MaxValueValidator(1)])


    def __str__(self):
        return self.text
