from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator

# Create your models here.
class EvalPredFile(models.Model):
    '''
        Django Model for Uploading predicted files
    '''
    truth_file = models.FileField(upload_to='evaluate/truth_file/')
    prediction_file = models.FileField(upload_to='evaluate/predict_file/')

class EvalAddFile(models.Model):
    '''
        Django Model for Uploading Addition Predicted files
    '''
    addition_pred_file = models.FileField(upload_to='evaluate/addition_pred_file/')

class EvalPretrainFile(models.Model):
    '''
        Django Model for Uploading pretrain file and the model requirement to Coco
    '''
    pretrain_file = models.FileField(upload_to='evaluate/pretrain_file/')

    description = models.TextField(default="getting-started")

    # doc default 120, min 120
    max_length = models.IntegerField(
        default=10,
        validators=[MaxValueValidator(512), MinValueValidator(10)]
    )
    # doc default 15, min 2
    epochs = models.IntegerField(
        default=15,
        validators=[MaxValueValidator(30), MinValueValidator(2)]
    )
    n_classes = models.IntegerField(
        default=2,
        validators=[MinValueValidator(2)]
    )

    max_features = models.IntegerField(
        default=100,
        validators=[MinValueValidator(10)]
    )

    embedding_size = models.IntegerField(
        default=25,
        validators=[MinValueValidator(10)]
    )



    TOKEN_CHOICE = (('word', 'word'),
                    ('char', 'char'),
                    ('transformer', 'transformer'))
    tokenization = models.CharField(default='char', max_length=20, choices=TOKEN_CHOICE)

    ARCH_CHOICE = (("cnn_char", "cnn_char"),
                   ("embed_bilstm_attend", "embed_bilstm_attend"),
                   ('transformer', 'transformer'))
    architecture = models.CharField(default='cnn_char', max_length=50, choices=ARCH_CHOICE)

    MODEL_CHOICE = (("bert-base-uncased", "bert-base-uncased"),
                    ("bert-base-cased", "bert-base-cased"),
                    ("bert-multilingual-cased", "bert-multilingual-cased"),
                    ("bert-base-german-cased", "bert-base-german-cased"),
                    )

    # Only needs for pretrain_model for BERT
    # pretrained_model = models.CharField(default="", max_length=50, choices=MODEL_CHOICE)

    finetune = models.BooleanField()
