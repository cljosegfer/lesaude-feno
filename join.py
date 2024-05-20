
# setup
import torch
import torch.nn as nn
import os

from configs.baseline import Downstream_cnn_args
from dataloaders.feno import CODE
from models.join import ResnetJoin
from runners.join import Runner


# init
database = CODE()

resnet_config = Downstream_cnn_args()
model = ResnetJoin(**resnet_config.__dict__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5
model_label = 'join'

runner = Runner(device, model, database, model_label)


# train
runner.train(EPOCHS)

# eval
runner.eval()