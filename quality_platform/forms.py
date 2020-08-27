from django import forms
from .models import EvalPredFile, EvalAddFile, EvalPretrainFile

class EvalFileForm(forms.ModelForm):
    '''
        Django Form for Uploading predicted files
    '''
    class Meta:
        model = EvalPredFile
        fields = ('truth_file', 'prediction_file')

class EvalAddFileForm(forms.ModelForm):
    '''
        Django Form for Uploading Addition Predicted files
    '''
    class Meta:
        model = EvalAddFile
        fields = ('addition_pred_file',)

class EvalPretrainForm(forms.ModelForm):
    '''
        Django Form for Uploading pretrain file and the model requirement to Coco
    '''
    class Meta:
        model = EvalPretrainFile
        fields = ('pretrain_file',
                  'description',
                  'max_features',
                  'max_length',
                  'embedding_size',
                  'epochs',
                  'n_classes',
                  'tokenization',
                  'architecture',
                  # 'pretrained_model',
                  'finetune',
                  )