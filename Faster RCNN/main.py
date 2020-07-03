import glob
import os
from os.path import basename

import torch

from checkpoint import load_ckp
from config import parameters
from dataloader_factory import get_custom_dataset
from models import pretrained_model
from train import train_fn


def main():
    os.makedirs('/'.join(parameters['checkpoint_path'].split('/')[:-1]), exist_ok=True)
    os.makedirs('/'.join(parameters['best_model_path'].split('/')[:-1]), exist_ok=True)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device : {device} is selected')
    file_names = [''.join(basename(i)[:-4]) for i in
                  glob.glob(os.path.join(parameters['image_dir'], '*'))]
    # print(file_names)
    train_loader, validation_loader = get_custom_dataset(file_names, parameters['train_frac'])

    model = pretrained_model('fasterrcnn_resnet50_fpn')
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=parameters['learning_rate'],
                                momentum=parameters['momentum'],
                                weight_decay=parameters['weight_decay'])
    try:
        model, optimizer, start_epoch = load_ckp(parameters['checkpoint_path'], model, optimizer)
    except Exception:
        print('No Previous Checkpoint Found....Training the model from scratch')
        start_epoch = 0
    best_loss = 1e10
    train_fn(start_epoch, parameters['epoch'], train_loader, validation_loader, model, device, optimizer, best_loss,parameters['checkpoint_path'],parameters['best_model_path'])

