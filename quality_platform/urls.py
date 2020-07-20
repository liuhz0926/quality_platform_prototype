from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='platform-home'),
    path('evaluate/', views.evaluate, name='platform-evaluate'),
    path('evaluate/upload-prediction/', views.eval_upload_prediction,
         name='platform-evaluate-upload-prediction'),
    # path('evaluate/load-prediction/', views.eval_load_prediction, name='platform-evaluate-load-prediction'),
    path('evaluate/report-prediction/', views.eval_report_prediction,
         name='platform-evaluate-report-prediction'),
    path('evaluate/report-prediction/confusion_matrix', views.eval_pred_report_confusion,
         name='platform-eval-pred-report-confusion'),
    path('evaluate/report-prediction/confusion_matrix/proportion', views.eval_pred_report_confusion_proportion,
         name='platform-eval-pred-report-confusion-proportion'),
    path('evaluate/report-prediction/threshold_analysis', views.eval_pred_report_threshold,
         name='platform-eval-pred-report-threshold'),
    path('evaluate/report-prediction/error_analysis', views.eval_pred_report_error,
         name='platform-eval-pred-report-error'),
    path('evaluate/report-prediction/upload_new', views.eval_pred_report_upload,
         name='platform-eval-pred-report-upload'),
]

