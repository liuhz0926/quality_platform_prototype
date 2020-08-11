from quality_platform.models import EvalPretrainFile
import requests
from .coco_config_toJson import config_to_python
from .get_labels import unzip, get_labels

HEADERS = {
    'accept': 'application/json',
    'SYMANTO_QP_ACCESS_TOKEN': 'QhV37mPdcz',
    'Content-Type': 'application/json',
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


    def post_train(self):
        print(self.data_model)
        print(self.dataset_url)
        params = (
            ('dataset_url', self.dataset_url),
        )

        response = requests.post('http://symanto-pastaepizza.northeurope.cloudapp.azure.com:8000/train',
                                 headers=HEADERS,
                                 params=params,
                                 data=self.data_model)

        print(response.json())