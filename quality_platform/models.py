from django.db import models

# Create your models here.
class EvalPredFile(models.Model):
    truth_file = models.FileField(upload_to='evaluate/truth_file/')
    prediction_file = models.FileField(upload_to='evaluate/predict_file/')