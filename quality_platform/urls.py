from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='platform-home'),

    # evaluation
    path('evaluate/', views.evaluate, name='platform-evaluate'),
    # for prediction files
    path('evaluate/prediction/', views.eval_upload_prediction,
         name='platform-evaluate-upload-prediction'),

    # reports
    path('report/evaluate/overview/', views.eval_report_overview,
         name='platform-evaluate-report'),
    path('report/evaluate/confusion_matrix/', views.eval_report_confusion,
         name='platform-eval-report-confusion'),
    path('report/evaluate/confusion_matrix/proportion', views.eval_report_confusion_proportion,
         name='platform-eval-report-confusion-proportion'),
    path('report/evaluate/threshold_analysis/', views.eval_report_threshold,
         name='platform-eval-report-threshold'),
    path('report/evaluate/error_analysis/', views.eval_report_error,
         name='platform-eval-report-error'),
    path('report/evaluate/upload_new/', views.eval_report_upload,
         name='platform-eval-report-upload-new'),

    # for pretrain model, use coco
    path('evaluate/pretrain/', views.eval_upload_pretrain,
         name='platform-evaluate-upload-pretrain'),
]

