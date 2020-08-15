from quality_platform.models import EvalPretrainFile
import requests
from .coco_config_toJson import config_to_python
from .get_labels import unzip, get_labels
import os
import time

HEADERS = {
    'accept': 'application/json',
    'SYMANTO_QP_ACCESS_TOKEN': 'QhV37mPdcz',
}


HOME_URL = "localhost:8000"
HOME_ADDRESS = '~/quality_platform_prototype/'

class Coco_request:
    def __init__(self):
        self.pretrain_form = EvalPretrainFile.objects.last()
        self.dataset_url = HOME_URL + self.pretrain_form.pretrain_file.url


        # get the labels from the train file
        zip_directory = HOME_ADDRESS + self.pretrain_form.pretrain_file.url
        unzip_directory = HOME_ADDRESS + 'uploads/evaluate/pretrain_file/'
        unzip(zip_directory, unzip_directory)
        train_labels = get_labels(unzip_directory + 'train.tsv')

        # HERE IS FOR THE SAMPLE TEST
        self.dataset_url = 'https://github.com/liuhz0926/quality_platform_prototype/raw/master/uploads/evaluate/pretrain_file/archive.zip'

        self.data_model = config_to_python(description=self.pretrain_form.description,
                                           dataset=self.pretrain_form.pretrain_file.name,
                                           tokenization=self.pretrain_form.tokenization,
                                           architecture=self.pretrain_form.architecture,
                                           pretrained_model=self.pretrain_form.pretrained_model,
                                           finetune=self.pretrain_form.finetune,
                                           max_length=self.pretrain_form.max_length,
                                           epochs=self.pretrain_form.epochs,
                                           n_classes=self.pretrain_form.n_classes,
                                           labels=train_labels
                                           )

    def request_coco(self):
        self.post_train()
        self.check_status()
        self.post_predict()

        return self.test_file, self.predict_file

    def post_train(self):
        headers = {
            'accept': 'application/json',
            'SYMANTO_QP_ACCESS_TOKEN': 'QhV37mPdcz',
            'Content-Type': 'application/json',
        }

        print(self.data_model)
        #print(self.dataset_url)
        params = (
            ('dataset_url', self.dataset_url),
        )

        response = requests.post(
            'http://symanto-pastaepizza.northeurope.cloudapp.azure.com:8000/train',
            headers=HEADERS,
            params=params,
            data=self.data_model
        )

        self.train_result = response.json()
        print(self.train_result)
        self.task_id = self.train_result["task_id"]
        self.model_name = self.train_result["model_name"]


    def get_status(self):
        headers = {
            'accept': 'application/json',
            'SYMANTO_QP_ACCESS_TOKEN': 'QhV37mPdcz',
        }
        params = {
            ('task_id', self.task_id),
        }

        response = requests.get(
            'http://symanto-pastaepizza.northeurope.cloudapp.azure.com:8000/status',
            headers=HEADERS,
            params=params
        )

        self.status = response.json()
        print(self.status)
        return self.status['status']


    def post_predict(self):

        self.test_file = os.path.expanduser(HOME_ADDRESS + 'uploads/evaluate/pretrain_file/test.tsv')

        url = 'http://symanto-pastaepizza.northeurope.cloudapp.azure.com:8000/predict'

        parms = {'model_name': self.model_name}

        with open(self.test_file, "rb") as f:
            response = requests.post(
                url=url,
                params=parms,
                headers=HEADERS,
                files={"dataset": ('text.tsv', f, "text/tsv")})

        self.predict_file = response.json()['url']


    def check_status(self):
        print('begin timer')
        timer1 = time.time()
        timer2 = time.time()
        finish_training = False

        while not finish_training:
            if timer2 - timer1 >= 2:
                print('two second and get status')
                status = self.get_status()
                print(status)
                if status.endswith("done."):
                    finish_training = True
                timer1 = timer2
                timer2 = time.time()
            else:
                timer2 = time.time()

        print('done')





