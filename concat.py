
# setup
import torch
import torch.nn as nn
import os

from configs.concat import Downstream_concat_args
from dataloaders.feno import CODE
from models.concat import ResnetConcat
from runners.concat import Runner


# init
database = CODE()

resnet_config = Downstream_concat_args()
model = ResnetConcat(**resnet_config.__dict__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5
model_label = 'concat'

runner = Runner(device, model, database, model_label)


# train
runner.train(EPOCHS)

# eval
runner.eval()