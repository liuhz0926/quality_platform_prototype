from django import forms
from .models import EvalPredFile, EvalAddFile

class EvalFileForm(forms.ModelForm):
    class Meta:
        model = EvalPredFile
        fields = ('truth_file', 'prediction_file')

class EvalAddFileForm(forms.ModelForm):
    class Meta:
        model = EvalAddFile
        fields = ('addition_pred_file',)