from django import forms
from .models import EvalPredFile

class EvalFileForm(forms.ModelForm):
    class Meta:
        model = EvalPredFile
        fields = ('truth_file', 'prediction_file')