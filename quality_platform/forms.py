from django import forms
from .models import EvalPredFile, EvalAddFile, EvalPretrainFile

class EvalFileForm(forms.ModelForm):
    class Meta:
        model = EvalPredFile
        fields = ('truth_file', 'prediction_file')

class EvalAddFileForm(forms.ModelForm):
    class Meta:
        model = EvalAddFile
        fields = ('addition_pred_file',)

class EvalPretrainForm(forms.ModelForm):
    class Meta:
        model = EvalPretrainFile
        fields = ('pretrain_file',
                  'tokenization',
                  'architecture',
                  'pretrained_model',
                  'finetune',
                  'max_length',
                  'epochs',
                  'n_classes',
                  )