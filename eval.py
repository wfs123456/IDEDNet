# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 11:10
# @Author  : Fusen Wang
# @Email   : 201924131014@cqu.edu.cn
# @File    : eval.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import dataset_constant_eval as Dataset
import torch.utils.data.dataloader as Dataloader
import torch
from torch.autograd import Variable
import os
import numpy as np
import tqdm

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def eval(model):
    model.eval()

    test_dataset = Dataset.Dataset(phase="test", gt_downsample=1)
    test_dataloader = Dataloader.DataLoader(test_dataset, batch_size=1, num_workers=0,
                                            shuffle=False, drop_last=False, worker_init_fn=worker_init_fn)
    with torch.no_grad():
        # mae = 0
        # mse = 0
        cc = 0.0
        cc_list = []
        for i,(images,dt_targets) in enumerate(tqdm.tqdm(test_dataloader)):
            images, dt_targets = Variable(images.cuda()), Variable(dt_targets.cuda())

            output = model(images)
            # print(output.size())
            # mae += abs(densitymaps.data.sum()-dt_targets.data.sum()).item()
            # mse += (densitymaps.data.sum()-dt_targets.data.sum()).item()**2
            # list_mae.append(abs(densitymaps.data.sum()-dt_targets.data.sum()).item())
            cc += CC(output.cpu().detach().numpy(), dt_targets.cpu().detach().numpy())
            cc_list.append(CC(output.cpu().detach().numpy(), dt_targets.cpu().detach().numpy()))
            # plt.imsave("./sample/output%s.png" % str(i + 1), output.cpu().squeeze(0).squeeze(0))

        # mae = mae / len(test_dataloader)
        # mse = (mse / len(test_dataloader)) **(1/2)
        cc = cc/len(test_dataloader)
    print( " cc: ",cc)
    with open("MOCS_Cifar_eval","w") as f:
        for index,i in enumerate(cc_list):
            f.write("index %d: "%(index+1)+str(i)+"\n")
        f.write("----------------------------------\n")
        f.write("cc: "+str(cc) + "\t")

def worker_init_fn(worker_id): # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import Unet

    model = Unet.Net()
    model = model.cuda()
    #weight_path = "models//Unet//MOCS_weights//Adam0.001_batch64//Unet_epoch_1869_mean_cc0.9341.pth"
    #weight_path = "models//Unet//HCM_weights//Adam0.001_batch64//Unet_epoch_1035_mean_cc0.9285.pth"
    #weight_path = "models//Unet//CCM_weights//Adam0.001_batch64//Unet_epoch_1972_mean_cc0.9456.pth"
    #weight_path = "models//Unet//Med_weights//Adam0.0001_batch64_L2//Unet_epoch_92_mean_cc0.9590.pth"
    # weight_path = "models//4.21_Unet//CCM_fashion_weights//Adam0.001_batch64_L1//Unet_epoch_293_mean_cc0.9210.pth"
    # weight_path = "models//4.21_Unet//HCM_fashion_weights//Adam0.001_batch64_L1//Unet_epoch_298_mean_cc0.9204.pth"
    # weight_path = "models//4.21_Unet//MOCS_fashion_weights//Adam0.001_batch64_L1//Unet_epoch_299_mean_cc0.9326.pth"
    weight_path = "models//4.21_Unet//MOCS_Cifar_weights//Adam0.001_batch64_L1//Unet_epoch_280_mean_cc0.8785.pth"
    print("weight path: %s\nloading weights..."%weight_path)
    weights = torch.load(weight_path)
    model.load_state_dict(weights)


    eval(model)



