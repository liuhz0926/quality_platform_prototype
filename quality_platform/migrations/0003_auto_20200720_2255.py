# Generated by Django 3.0.7 on 2020-07-20 22:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('quality_platform', '0002_auto_20200625_0108'),
    ]

    operations = [
        migrations.CreateModel(
            name='EvalAddFile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('addition_pred_file', models.FileField(upload_to='evaluate/addition_pred_file/')),
            ],
        ),
        migrations.AlterField(
            model_name='evalpredfile',
            name='prediction_file',
            field=models.FileField(upload_to='evaluate/predict_file/'),
        ),
        migrations.AlterField(
            model_name='evalpredfile',
            name='truth_file',
            field=models.FileField(upload_to='evaluate/truth_file/'),
        ),
    ]
