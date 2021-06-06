# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 10:06
# @Author  : Fusen Wang
# @Email   : 201924131014@cqu.edu.cn
# @File    : train.py
# @Software: PyCharm
from config import *
import Unet
import IDEDNet
import dataset_constant as Dataset
import torch.utils.data.dataloader as Dataloader
import torch.nn as nn
import torch.optim as optim
import time
import torch
import visdom
from torch.autograd import Variable
import sys
import numpy as np
import os
import random
import warnings

warnings.filterwarnings("ignore")

def CC(img, gtp):
    """
    cc
    :param img:
    :param gtp:
    :return:
    """
    o_avg = np.mean(img)
    p_avg = np.mean(gtp)
    # o_avg = img.mean()
    # p_avg = gtp.mean()
    ccz = np.sum(np.multiply(img - o_avg, gtp - p_avg))
    ccm = (np.sum((img - o_avg) ** 2) * np.sum((gtp - p_avg) ** 2)) ** 0.5
    cc = ccz / (ccm + 0.0000000000001)
    cc = np.abs(cc)
    # print("o_avg,p_avg", o_avg, p_avg)
    return cc

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA

#viz = visdom.Visdom(env=SAVE_PATH.replace("/","_"))

def train():

    # if not os.path.exists("models/%s/%s" % (SAVE_PATH.split("/")[0],SAVE_PATH.split("/")[1])):
    #     os.mkdir("models/%s/%s" % (SAVE_PATH.split("/")[0],SAVE_PATH.split("/")[1]))
    # if not os.path.exists("models/%s/%s/%s" % (SAVE_PATH.split("/")[0],SAVE_PATH.split("/")[1],SAVE_PATH.split("/")[2])):
    #     os.mkdir("models/%s/%s/%s" % (SAVE_PATH.split("/")[0],SAVE_PATH.split("/")[1],SAVE_PATH.split("/")[2]))
    if not os.path.exists("models/%s"%SAVE_PATH):
        os.makedirs("models/%s"%SAVE_PATH)


    config_log = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()) + \
        "\n-------------------------------------------------------------" \
        "\nconfig:\n%s" \
        "-------------------------------------------------------------"
    l_temp = ""
    for i in range(len(VAR_LIST)):
        l_temp += "\t%s\n" % VAR_LIST[i]
    config_log = config_log % l_temp
    with open(os.path.join("models", SAVE_PATH, "log.txt"), "a+") as f:
        f.write(config_log + "\n\n")
    print(config_log)

    dataset = Dataset.Dataset(gt_downsample=1)
    dataloader = Dataloader.DataLoader(dataset, batch_size=BATCH_SIZE,num_workers=0,
                        shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)
    # print("dataset size is: %d"%dataset.__len__())

    test_dataset = Dataset.Dataset(phase="test", gt_downsample=1)
    test_dataloader = Dataloader.DataLoader(test_dataset,batch_size=1,num_workers=0,
                        shuffle=False, drop_last=True, worker_init_fn=worker_init_fn)

    net = Unet.Net()
    net = net.cuda()
    if OPTIMIZER == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    if LOSS_F == "MSE":
        criterion = nn.MSELoss(reduction='sum').cuda()
    elif LOSS_F == "L1":
        criterion = nn.L1Loss(reduction='sum').cuda()

    t0 = time.time()
    start_epoch = 0
    step_index = 0

    min_mae = 1.0
    min_epoch = -1
    epoch_list = []
    train_loss_list = []
    epoch_loss_list = []
    test_CC_list = []


    for i in range(start_epoch, MAX_EPOCH):

        if LR_DECAY and (i in STEPS):
            adjust_learning_rate(optimizer, LR_DECAY)

        ## train ##
        epoch_loss = 0
        epoch_cc = 0
        net.train()
        for _,(images,dt_targets) in enumerate(dataloader):

            images,dt_targets = Variable(images.cuda()),Variable(dt_targets.cuda())

            output = net(images)
            # print('a,b,c',images.size(),output.size(),dt_targets.size())
            loss = criterion(output, dt_targets) / BATCH_SIZE
            cc = CC(output.cpu().detach().numpy(), dt_targets.cpu().detach().numpy())
            epoch_cc += cc.item()
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss_list.append(epoch_loss)
        train_loss_list.append(epoch_loss/len(dataloader))
        epoch_list.append(i+1)
        localdate = time.strftime("%Y/%m/%d %H:%M:%S",time.localtime())
        with open(os.path.join("models",SAVE_PATH,"log.txt"),"a+") as f:
            f.write(localdate+"\n")
        print(localdate)
        train_log = "train [%d/%d] timer %.4f, loss %.4f, cc %.4f" % \
                    (i, MAX_EPOCH, time.time() - t0, epoch_loss / len(dataloader), epoch_cc/len(dataloader))
        with open(os.path.join("models",SAVE_PATH,"log.txt"),"a+") as f:
            f.write(train_log+"\n")
        print(train_log)

        t0 = time.time()

        ## eval ##
        net.eval()
        with torch.no_grad():
            test_cc = 0.0
            epoch_cc_list = []
            # mae = 0
            # mse = 0

            for _,(images,dt_targets) in enumerate(test_dataloader):
                images, dt_targets = Variable(images.cuda()), Variable(dt_targets.cuda())

                output = net(images)
                # print('a,b,c', images.size(), output.size(), dt_targets.size())
                cc_ = CC(output.cpu().detach().numpy(),dt_targets.cpu().detach().numpy())
                epoch_cc_list.append(cc_)
                test_cc += cc_
                # TODO bug?
                # mae += abs(densitymaps.data.sum()-dt_targets.data.sum()).item()
                # mse += (densitymaps.data.sum()-dt_targets.data.sum()).item()**2

            test_cc = test_cc / len(test_dataloader)
            # mae = mae / len(test_dataloader)
            # mse = (mse / len(test_dataloader)) **(1/2)
            test_CC_list.append(test_cc)
            if(1-test_cc<min_mae):
                min_mae = 1-test_cc
                min_epoch = i
                save_log = "save state, epoch: %d" % i
                with open(os.path.join("models", SAVE_PATH, "log.txt"), "a+") as f:
                    f.write(save_log + "\n")
                print(save_log)
                torch.save(net.state_dict(), "models/%s/%s_epoch_%d_mean_cc%.4f.pth" % (SAVE_PATH,MODEL,i,test_cc))
            # test_mae_list.append(mae)

            eval_log = "eval [%d/%d] , mean_cc %.4f, min_epoch %d\n"%(i,MAX_EPOCH,test_cc, min_epoch)
            with open(os.path.join("models",SAVE_PATH,"log.txt"),"a+") as f:
                f.write(eval_log+"\n")
            print(eval_log + '\n' + str(epoch_cc_list))

            # vis ##
            # viz.line(win="1", X=epoch_list, Y=train_loss_list,
            #          opts=dict(legend = [MODEL], xlabel = 'Epoch', ylabel = 'L1loss' ,title="Train_loss"))
            # viz.line(win="2", X=epoch_list, Y=test_CC_list,
            #          opts=dict(legend = [MODEL], xlabel = 'Epoch', ylabel = 'Corr' ,title="Test_Correlation coefficient"))
            #
            # index = random.randint(0,len(test_dataloader)-1)
            # image,gt_map = test_dataset[index]
            # viz.image(win="3",img=image,opts=dict(title="test_image"))
            # viz.image(win="4",img=gt_map/(gt_map.max())*255,opts=dict(title="gt_map_%.4f"%(gt_map.sum())))
            #
            # image = Variable(image.unsqueeze(0).cuda())
            # # densitymap,_ = net(image)
            # densitymap = net(image)
            # densitymap = densitymap.squeeze(0).detach().cpu().numpy()
            # viz.image(win="5",img=densitymap/(densitymap.max())*255,opts=dict(title="predictImages_%.4f"%(densitymap.sum())))

def adjust_learning_rate(optimizer, gamma):
    lr = LEARNING_RATE * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def setup_seed(seed=19960715):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed) #gpu
    np.random.seed(seed) #numpy
    random.seed(seed)
    torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id): # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == "__main__":
    setup_seed()
    train()