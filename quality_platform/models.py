from django.db import models

# Create your models here.
class EvalPredFile(models.Model):
    truth_file = models.FileField(upload_to='evaluate/truth_file/')
    prediction_file = models.FileField(upload_to='evaluate/predict_file/')

class EvalAddFile(models.Model):
    addition_pred_file = models.FileField(upload_to='evaluate/addition_pred_file/')

class EvalPretrainFile(models.Model):
    pretrain_file = models.FileField(upload_to='evaluate/pretrain_file/')

    TOKEN_CHOICE = (('word', 'word'),
                    ('char', 'char'),
                    ('transformer', 'transformer'))
    tokenization = models.CharField(max_length=20, choices=TOKEN_CHOICE)

    ARCH_CHOICE = (("cnn_char", "cnn_char"),
                   ("embed_bilstm_attend", "embed_bilstm_attend"),
                   ('transformer', 'transformer'))
    architecture = models.CharField(max_length=50, choices=ARCH_CHOICE)

    MODEL_CHOICE = (("bert-base-german-cased", "bert-base-german-cased"),
                    ("bert-base-cased", "bert-base-cased"),
                    ("bert-multilingual-cased", "bert-multilingual-cased"),
                    ("bert-base-german-cased", "bert-base-german-cased"))
    pretrained_model = models.CharField(max_length=50, choices=MODEL_CHOICE)

    finetune = models.BooleanField()
    max_length = models.IntegerField()
    epochs = models.IntegerField()
    n_classes = models.IntegerField()