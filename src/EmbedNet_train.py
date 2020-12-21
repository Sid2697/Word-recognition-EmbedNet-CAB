"""
This file is used for training a EmbedNet for word recognition
"""

# Standard library imports
import os
import time
import argparse

# Third party imports
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from models import EmbedNet
from triplets import Triplets
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(
    description='Neural Networks for word recognition')
# File paths and directory names
parser.add_argument(
    '--base_dir',
    default='trained',
    help='Path to the directory for saving models')
parser.add_argument(
    '--model_dir',
    default='EmbedNet_models',
    help='Name of the folder for saving trained models')

# Various model hyperparameters
parser.add_argument(
    '--train_percentage',
    type=float,
    default=0.8,
    help='Percentage of data to use for training')
parser.add_argument(
    '--epochs',
    type=int,
    default=1000,
    help='Number of epochs')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    help='Learning Rate')
parser.add_argument(
    '--batch',
    type=int,
    default=32,
    help='Batch size')
parser.add_argument(
    '--model_name',
    help='Name of the model for saving (Naming scheme: WNet{num. of layers}\
        {Optimiser Name}LR{learning rate}EXT{Some other information})')
parser.add_argument(
    '--margin',
    type=float,
    default=1,
    help='Triplet Loss margin')
parser.add_argument(
    '--hidden_layers',
    nargs='+',
    type=int,
    default=[1024, 512, 256, 128],
    help='List of input size of the hidden layers')

parser.add_argument(
    '--gpu_id',
    default=0,
    type=int,
    help='Specify which GPU to use')
parser.add_argument(
    '--image_embeds',
    default='embeddings/topk_preds_100featsImg.npy',
    help='Path to the image embeddings')
parser.add_argument(
    '--topk_embeds',
    default='embeddings/topk_preds_100featsSynth.npy',
    help='Path to the text embeddigns')
parser.add_argument(
    '--image_file',
    default='gen_files/image_embed_top_k_100.txt',
    help='Path to the file containing word image information')
parser.add_argument(
    '--predictions_file',
    default='gen_files/top_preds_embeds_100_with_conf.txt',
    help='Path to the file containing text output information')
args = parser.parse_args()
print(args)

print('[INFO] Loading embeddings and text files...')
image_embeds = np.load(args.image_embeds)
try:
    topk_embeds = np.load(args.topk_embeds)
except Exception as e:
    print('[INFO] Error: {} Loading text embeddings \
        in memmap mode...'.format(e))
    topk_embeds = np.memmap(
        args.topk_embeds,
        dtype=np.float32,
        mode='r',
        shape=(2109500, 2048))

with open(args.image_file, 'r') as image_file:
    image_file = image_file.readlines()
image_file = [item.split()[1] for item in image_file]

with open(args.predictions_file, 'r') as text_file:
    predictions_file = text_file.readlines()
topk_info = [item.split()[1] for item in predictions_file]

assert args.model_name, "Provide a model name for proceeding"
epochs = args.epochs
lr = args.lr
model_dir = args.model_dir
if not os.path.exists(os.path.join(args.base_dir, model_dir)):
    os.makedirs(os.path.join(args.base_dir, model_dir))

if torch.cuda.device_count() > 1:
    torch.cuda.set_device(args.gpu_id)
    print('[INFO] Using GPU with ID={}'.format(args.gpu_id))


def save_checkpoint(
        model_path,
        epoch,
        model,
        optimizer,
        hard=False,
        temp=False,
        accuracy=False):
    """Save the checkpoint."""
    state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }
    if hard:
        print('[INFO] Saving hard negatives model...')
        torch.save(state, os.path.join(
            model_path,
            args.model_name + '_best_hard.pkl'))
    elif temp:
        print('[INFO] Saving temporary model...')
        torch.save(state, os.path.join(
            model_path,
            args.model_name + '_temp.pkl'))
    elif accuracy:
        print('[INFO] Saving a better accuracy model...')
        torch.save(state, os.path.join(
            model_path,
            args.model_name + '_accuracy.pkl'))
    else:
        print('[INFO] Saving model...')
        torch.save(state, os.path.join(model_path, args.model_name + '.pkl'))
    with open(os.path.join(model_path, args.model_name + '.txt'), 'a') \
            as model_file:
        model_file.write('\n=====\nTime of saving: {}\n'.format(time.time()))
        model_file.write('\n=====\nModel: {}\n'.format(str(model)))
        model_file.write('\n=====\nEpoch: {}\n'.format(epoch))


def get_dataloaders(train_list):
    """Generate the train and val list"""
    train_count = int(len(train_list)*args.train_percentage)
    val_count = len(train_list) - train_count
    train_data = train_list[:train_count]
    val_data = train_list[-val_count:]
    train_data_loader = DataLoader(
        dataset=train_data,
        num_workers=1,
        batch_size=args.batch,
        shuffle=False,
        pin_memory=True)
    val_data_loader = DataLoader(
        dataset=val_data,
        num_workers=1,
        batch_size=args.batch,
        shuffle=False,
        pin_memory=True)
    del train_data
    del val_data
    return train_data_loader, val_data_loader


triplet = Triplets(
    topk_info,
    image_file,
    topk_embeds,
    image_embeds,
    args.train_percentage,
    args.margin,
    verbose=True)
train_list = triplet.initial_list()

train_data_loader, val_data_loader = get_dataloaders(train_list)
del train_list

model = EmbedNet(2048, 128, hidden_layers=args.hidden_layers)
print('[INFO] EmbedNet Architecture:\n{}'.format(model))

criterion = nn.TripletMarginLoss(margin=args.margin, p=2)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

if os.path.exists(
        os.path.join(
        os.path.join(args.base_dir, model_dir),
        args.model_name + '.pkl')):
    print('[INFO] Loading a previously saved checkpoint...')
    checkpoint = torch.load(os.path.join(
        os.path.join(args.base_dir, model_dir),
        args.model_name + '.pkl'))
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # Solves the issue of optimiser loading on the CPU
    # (https://github.com/pytorch/pytorch/issues/2830)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                if torch.cuda.is_available():
                    state[k] = v.cuda().double()
                else:
                    state[k] = v.double()

train_batch_count = 0
validation_batch_count = 0
base_valid = np.inf
model_saved_epoch = 0
old_hard_neg_number = np.inf
accuracy = 0

for epoch in range(epochs):
    start = time.time()
    train_loss_per_epoch = 0
    val_loss_per_epoch = 0
    for data_point in tqdm(
            train_data_loader,
            desc='[INFO] Epoch {}/{}'.format(epoch, epochs)):
        train_batch_count += 1
        anchor = data_point['anchor']
        positive = data_point['positive']
        negative = data_point['negative']
        model.train()
        if torch.cuda.is_available():
            model = model.cuda().double()
            anchor = anchor.cuda().double()
            positive = positive.cuda().double()
            negative = negative.cuda().double()
        else:
            model = model.double()
        model.zero_grad()
        anchor_ = model(anchor)
        positive_ = model(positive)
        negative_ = model(negative)
        tr_loss = criterion(anchor_, positive_, negative_)
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()
        train_loss_per_epoch += float(tr_loss)
    for data_point in tqdm(val_data_loader, desc='[INFO] Validation'):
        validation_batch_count += 1
        anchor = data_point['anchor']
        positive = data_point['positive']
        negative = data_point['negative']
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda().double()
            anchor = anchor.cuda().double()
            positive = positive.cuda().double()
            negative = negative.cuda().double()
        anchor_ = model(anchor)
        positive_ = model(positive)
        negative_ = model(negative)
        val_loss = criterion(anchor_, positive_, negative_)
        val_loss_per_epoch += float(val_loss)
    if val_loss_per_epoch < base_valid:
        base_valid = val_loss_per_epoch
        save_checkpoint(
            os.path.join(args.base_dir, model_dir),
            epoch,
            model,
            optimizer)
    print('[INFO] Train Loss {}, validation loss '
          '{}.'.format(
            round(train_loss_per_epoch, 3),
            round(val_loss_per_epoch, 3)))
    if (epoch + 1) - model_saved_epoch >= 5:
        print('[INFO] Updating the train and validation list...')
        updated_list, new_hard_neg_number = triplet.embednet_embeds(model, 128)
        if new_hard_neg_number < old_hard_neg_number:
            save_checkpoint(
                os.path.join(args.base_dir, model_dir),
                epoch,
                model,
                optimizer,
                hard=True)
            old_hard_neg_number = new_hard_neg_number
        train_data_loader, val_data_loader = get_dataloaders(updated_list)
        model_saved_epoch = epoch + 1
