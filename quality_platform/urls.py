from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='platform-home'),

    # evaluation
    path('evaluate/', views.evaluate, name='platform-evaluate'),
    # for prediction files
    path('evaluate/prediction/', views.eval_upload_prediction,
         name='platform-evaluate-upload-prediction'),
    path('report/evaluate/prediction/overview/', views.eval_report_prediction,
         name='platform-evaluate-report-prediction'),
    path('report/evaluate/prediction/confusion_matrix/', views.eval_pred_report_confusion,
         name='platform-eval-pred-report-confusion'),
    path('report/evaluate/prediction/confusion_matrix/proportion', views.eval_pred_report_confusion_proportion,
         name='platform-eval-pred-report-confusion-proportion'),
    path('report/evaluate/prediction/threshold_analysis/', views.eval_pred_report_threshold,
         name='platform-eval-pred-report-threshold'),
    path('report/evaluate/prediction/error_analysis/', views.eval_pred_report_error,
         name='platform-eval-pred-report-error'),
    path('report/evaluate/prediction/upload_new/', views.eval_pred_report_upload,
         name='platform-eval-pred-report-upload'),

    # for pretrain model, use coco
    path('evaluate/pretrain/', views.eval_upload_pretrain,
         name='platform-evaluate-upload-pretrain'),
]

