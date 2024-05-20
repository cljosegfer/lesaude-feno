
# # setup
import torch
import torch.nn as nn
import os

from configs.baseline import Downstream_cnn_args
from dataloaders.baseline import CODE
from models.baseline import ResnetBaseline
from runners.baseline import Runner
from utils import load_backbone


# # init
database = CODE()

resnet_config = Downstream_cnn_args()
model = ResnetBaseline(**resnet_config.__dict__)
model = load_backbone(model, 'output/pretrain/pretrain.pt')['model']
# model = torch.load('output/finetuning/partial_0.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5
model_label = 'finetuning'

runner = Runner(device, model, database, model_label)


# # train
runner.train(EPOCHS)


# # eval
runner.eval()
