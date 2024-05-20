
# # setup
import torch
import torch.nn as nn
import os

from configs.pretrain import Downstream_cnn_args
from dataloaders.feno import CODE
from models.baseline import ResnetBaseline
from runners.pretrain import Runner


# # init
database = CODE()

resnet_config = Downstream_cnn_args()
model = ResnetBaseline(**resnet_config.__dict__)
model = torch.load('output/pretrain/partial_1.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5
model_label = 'pretrain'

runner = Runner(device, model, database, model_label)


# # train
runner.train(EPOCHS)
