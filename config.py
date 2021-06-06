# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 11:16
# @Author  : Fusen Wang
# @Email   : 201924131014@cqu.edu.cn
# @File    : config.py
# @Software: PyCharm

HOME = "D:\\research\\image\\"
DATASET = "CCM_Cifar" #chaotic_mnist,Threemap_Cifar10, pascal, CCM_mnist，MOCS_mnist,HCM_mnist,Medmnist

CUDA = "0"

RESUME = False

BATCH_SIZE = 64
MOMENTUM = 0.95
WEIGHT_DECAY =5*1e-4   #5*1e-4
LEARNING_RATE = 1e-4
MAX_EPOCH = 300
STEPS = (i for i in range(0,10,MAX_EPOCH))
LR_DECAY = 0.92  # 0.95
OPTIMIZER = "Adam"
LOSS_F = "L1"

MODEL = "Unet" ##FCN,UNet,MSEDNet,IDEDNet

SAVE_PATH = "4.21_%s/%s_weights/%s%s_batch%s_L1"%(MODEL, DATASET, OPTIMIZER,str(LEARNING_RATE),str(BATCH_SIZE))

RANDOM_CROP = False
RANDOM_HFLIP = 0.4
RANDOM_VFLIP = 0.4        # 0.5
RANDOM_2GRAY = False       # 0.2
DIVIDE = False             # 16



VAR_LIST = ["BATCH: %d"%BATCH_SIZE, "OPTIM: %s"%OPTIMIZER, "LR: %s"%str(LEARNING_RATE), "LOSS_F: %s"%LOSS_F,
            "CUDA: %s"%CUDA,"LR_DECAY: %s"%LR_DECAY, "MODEL: %s"%MODEL, "RANDOM_CROP: %s"%str(RANDOM_CROP),
            "RANDOM_HFLIP: %s"%str(RANDOM_HFLIP), "RANDOM_VFLIP: %s"%str(RANDOM_VFLIP),"RANDOM_2GRAY: %s"%str(RANDOM_2GRAY),"DIVIDE: %s"%DIVIDE,
            "SAVE_PATH: %s"%SAVE_PATH, "DATASET: %s"%DATASET]
