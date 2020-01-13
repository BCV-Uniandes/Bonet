# -*- coding: utf-8 -*-

"""
Bone Age Assessment BoNet test routine.
"""

# Standard lib imports
import os
import csv
import glob
import time
import argparse
import warnings
import pandas as pd
import os.path as osp

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import horovod.torch as hvd
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Local imports
from models.bonet import BoNet
from data.boneage_loader import BoneageDataset

# Other imports
from tqdm import tqdm
import pdb

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# Dataloading-related settings
parser.add_argument('--heatmaps', default=False, action='store_true',
                help='Test model with gaussian heatmaps')
parser.add_argument('--cropped', default=False, action='store_true',
                help='Test model with cropped images according to bbox')
parser.add_argument('--dataset', default='RSNA', type=str,choices=['RSNA','RHPE'],
                help='Dataset to perform test')

# Dataloading-related settings
parser.add_argument('--data-test', default='data/test/', type=str,
                help='path to test data folder')
parser.add_argument('--ann-path-test', default='test.csv', type=str,
                help='path to BAA annotations file')
parser.add_argument('--rois-path-test', default='test.json',
                type=str, help='path to ROIs annotations in coco format')

parser.add_argument('--save-folder', default='TRAIN/new_test/',
                help='location to save checkpoint models')
parser.add_argument('--snapshot', default='boneage_bonet_weights.pth',
                help='path to weight snapshot file')


parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                help='number of data loading workers (default: 4)')

# Training procedure settings
parser.add_argument('--batch-size', default=1, type=int,
                help='Batch size for training')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--gpu', type=str, default='2,3')


args = parser.parse_args()

args_dict = vars(args)
print('Argument list to program')
print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                 for arg in args_dict]))
print('\n\n')

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if not os.path.exists(os.path.join(args.save_folder, 'inference')):
    os.makedirs(os.path.join(args.save_folder, 'inference'))


# Horovod settings
hvd.init()
torch.cuda.set_device(hvd.local_rank())
torch.cuda.manual_seed(hvd.size())

args.distributed = hvd.size() > 1
args.rank = hvd.rank()
args.size = hvd.size()

# CREATE THE NETWORK ARCHITECTURE AND LOAD THE BEST MODEL
if args.heatmaps:
    from models.bonet_heatmap import BoNet
else:
    from models.bonet import BoNet

net = BoNet()

if args.rank == 0:
    print('---> Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))

if osp.exists(args.snapshot):
    model_to_load=args.snapshot
else:
    model_to_load=args.save_folder+'/'+args.snapshot

if osp.exists(model_to_load) and args.rank == 0:
    print('Loading state dict from: {0}'.format(model_to_load))
    snapshot_dict = torch.load(model_to_load, map_location=lambda storage, loc: storage)
    weights= net.state_dict()
    new_snapshot_dict=snapshot_dict.copy()
    for key in snapshot_dict:
        if key not in weights.keys():
            new_key='inception_v3.'+key
            new_snapshot_dict[new_key]=snapshot_dict[key]
            new_snapshot_dict.pop(key)

    net.load_state_dict(new_snapshot_dict)

net = net.to(device)

# Criterion
criterion = nn.L1Loss()

# Horovod
hvd.broadcast_parameters(net.state_dict(), root_rank=0)

# Dataloader
test_transform = transforms.Compose([transforms.Resize((500, 500)),
                               transforms.ToTensor()])

if args.heatmaps:
    from data.data_loader import Boneage_HeatmapDataset as Dataset
else:
    from data.data_loader import BoneageDataset as Dataset

test_dataset = Dataset(args.data_test, args.ann_path_test,args.rois_path_test,
                                  img_transform=test_transform,crop=args.cropped,dataset=args.dataset)

# Data samplers
test_sampler = None

if args.distributed:
    test_sampler = DistributedSampler(test_dataset,
                                      num_replicas=args.size,
                                      rank=args.rank)

test_loader = DataLoader(test_dataset,
                        shuffle=False, 
                        sampler=test_sampler,
                        batch_size=1,
                        num_workers=args.workers)

def main():
    print('Inference begins...')
    carpograms = pd.read_csv(os.path.join('Paths', args.ann_path_test))
    ids = carpograms.ix[:, 0]
    p_dict = dict.fromkeys(ids)
    p_dict = test(args, net, test_loader, test_sampler,
                  criterion, p_dict)
    df = pd.DataFrame.from_dict(p_dict, orient="index")
    df.to_csv(os.path.join(args.save_folder, 'test.csv'))



def evaluate():
    net.eval()
    epoch_total_loss = AverageMeter()
    for (batch_idx, (imgs, bone_ages, genders, _)) in enumerate(test_loader):
        imgs = imgs.to(device)
        bone_ages = bone_ages.to(device)
        genders = genders.to(device)
        with torch.no_grad():
            outputs = net(imgs, genders)
        loss = criterion(outputs.squeeze(), bone_ages)
        loss = metric_average(loss.item(), 'loss')
        epoch_total_loss.update(loss, 1)

    epoch_total_loss = epoch_total_loss.avg

    if args.rank == 0:
        print('Val loss: {:.5f}'.format(epoch_total_loss))

    return epoch_total_loss

def test(args, net, loader, sampler, criterion, p_dict):
    net.eval()
    epoch_loss = AverageMeter()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader, 0)):
            inputs, labels, gender, p_id = batch
            inputs, gender = Variable(inputs).cuda(), Variable(gender).cuda()
            labels = Variable(labels).cuda()
            outputs = net(inputs, gender)

            p_dict[p_id] = outputs
            loss = criterion(outputs.squeeze_(), labels)
            
            epoch_loss.update(loss)
    loss = metric_average(epoch_loss.avg,'loss')

    if args.rank == 0:
        print('Test loss: {}'.format(loss))
    return p_dict


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
