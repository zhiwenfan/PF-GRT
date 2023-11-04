import os, sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from tqdm import trange
from test import val
from core.dataset import LINEMOD_SO3 as LINEMOD
from core.loss import weighted_infoNCE_loss_func
from core.utils import load_checkpoint
from core.model import RetrievalNet as Model
from core.model import Sim_predictor as Predictor

np.set_printoptions(threshold=np.inf)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)
np.random.seed(0)

def train_one_epoch(epoch, train_loader, model, predictor, optimizer):
    model.train()
    predictor.train()
    train_loss = []
    train_errs = []
    for i, data in enumerate(train_loader):
        ## load data and label
        src_img, ref_img, src_R, ref_R, id = data

        if torch.any(id == -1):
            print("Skip incorrect data")
            continue

        src_img = src_img.cuda()
        ref_img = torch.cat(ref_img, dim=0).cuda()
        src_R, ref_R = src_R.cuda(), ref_R.cuda()
        id = id.cuda()

        B, _, H, W = src_img.shape

        ## feature extraction
        src_f = model(src_img)
        ref_f = model(ref_img)

        ## similarity estimation
        ref_sim = predictor(src_f, ref_f)

        ## loss estimation
        loss = weighted_infoNCE_loss_func(ref_sim[:, B:2*B], ref_sim[:, 2*B:], ref_R[:, 1], ref_R[:, 2], src_R, id, tau=0.1)

        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            print("Skip incorrect data")
            continue

        train_loss.append(loss.item())

        if i % 20 == 0:
            print("\tEpoch %3d --- Iter [%d/%d] Train --- Loss: %.4f" % (epoch, i + 1, len(train_loader), loss.item()))

    train_loss = np.asarray(train_loss).mean()
    return train_loss

def train(cfg, device):
    print(">>>>>>>>>>>>>> CREATE DATASET")
    dataset_train = LINEMOD(cfg, 'train', 0)
    train_loader = DataLoader(dataset_train, batch_size=cfg["TRAIN"]["BS"], shuffle=True, \
    num_workers=cfg["TRAIN"]["WORKERS"], drop_last=True)
    print(">>>>>>>>>> TRAINING DATA:", len(train_loader)*cfg["TRAIN"]["BS"])

    print(">>>>>>>>>>>>>> CREATE NETWORK")
    model = Model(cfg).to(device)
    predictor = Predictor(cfg).to(device)

    print(">>>>>>>>>>>>>> CREATE OPTIMIZER")
    optimizer = optim.Adam([{'params': model.parameters()}, {'params': predictor.parameters()}], lr=cfg["TRAIN"]["LR"])
    lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg["TRAIN"]["STEP"], gamma=cfg["TRAIN"]["GAMMA"])

    if not os.path.exists(cfg["TRAIN"]["WORKING_DIR"]):
        os.makedirs(cfg["TRAIN"]["WORKING_DIR"])

    logname = os.path.join(cfg["TRAIN"]["WORKING_DIR"], 'training_log.txt')
    with open(logname, 'a') as f:
        f.write('training set: ' + str(len(dataset_train)) + '\n')

    if cfg["TRAIN"]["FROM_SCRATCH"] is False:
        print(">>>>>>>>>>>>>> LOAD MODEL")
        model, optimizer, start_epoch, best_acc = load_checkpoint(model, optimizer, cfg["TRAIN"]["WORKING_DIR"] + "checkpoint_1.pth")
        predictor, _, _, _ = load_checkpoint(predictor, optimizer, cfg["TRAIN"]["WORKING_DIR"] + "checkpoint_2.pth")
    else:
        print(">>>>>>>>>>>>>> TRAINING FROM SCRATCH")
        best_acc = 0
        start_epoch = 0

    print(">>>>>>>>>>>>>> START TRAINING")
    for epoch in trange(start_epoch, cfg["TRAIN"]["MAX_EPOCH"]):
        loss = train_one_epoch(epoch, train_loader, model, predictor, optimizer)

        # update learning rate
        lrScheduler.step()

        if (epoch + 1) % cfg["TRAIN"]["VAL_STEP"] == 0:
            res = val(cfg, model, predictor, device)

            if res[0] > best_acc:
                best_acc = res[0]
                state_dict = {'epoch': epoch, 'state_dict': model.state_dict(),\
                    'optimizer': optimizer.state_dict(),
                    'best_acc': res[0]}
                torch.save(state_dict, os.path.join(cfg["TRAIN"]["WORKING_DIR"], 'checkpoint_1.pth'))

                state_dict = {'epoch': epoch, 'state_dict': predictor.state_dict(),\
                    'optimizer': optimizer.state_dict(),
                    'best_acc': res[0]}
                torch.save(state_dict, os.path.join(cfg["TRAIN"]["WORKING_DIR"], 'checkpoint_2.pth'))

            with open(logname, 'a') as f:
                text = str('Epoch: %03d || train_loss %.4f || test_cls_acc: %.2f || test_R_acc %.2f\n' % (epoch, loss, res[1], res[0]))
                f.write(text)
        else:
            with open(logname, 'a') as f:
                text = str('Epoch: %03d || train_loss %.4f \n' % (epoch, loss))
                f.write(text)

if __name__ == '__main__':
    print(">>>>>>>>>>>>> Loding configuration")
    with open("./objects.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    train(cfg, device)
