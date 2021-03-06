# Generated by Django 3.0.7 on 2020-07-30 04:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('quality_platform', '0003_auto_20200720_2255'),
    ]

    operations = [
        migrations.CreateModel(
            name='EvalPretrainFile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pretrain_file', models.FileField(upload_to='evaluate/pretrain_file/')),
                ('tokenization', models.CharField(choices=[('word', 'word'), ('char', 'char'), ('transformer', 'transformer')], max_length=20)),
                ('architecture', models.CharField(choices=[('cnn_char', 'cnn_char'), ('embed_bilstm_attend', 'embed_bilstm_attend'), ('transformer', 'transformer')], max_length=50)),
                ('pretrained_model', models.CharField(choices=[('bert-base-german-cased', 'bert-base-german-cased'), ('bert-base-cased', 'bert-base-cased'), ('bert-multilingual-cased', 'bert-multilingual-cased'), ('bert-base-german-cased', 'bert-base-german-cased')], max_length=50)),
                ('finetune', models.BooleanField()),
                ('max_length', models.IntegerField()),
                ('epochs', models.IntegerField()),
                ('n_classes', models.IntegerField()),
            ],
        ),
    ]
