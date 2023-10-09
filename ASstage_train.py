# import libraries
import os

import time
import logging
import random
import torch

import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import get_data, normalization, TraindataSet, randomTransform
from network import VINet, ASNet
from LossVGG import VGGLoss

# -------------------------------------#
# ---------Initialization--------------#
# -------------------------------------#

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Set GPU device
gpu_list = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set directories
# data directory
inputSax_dir = "./data/Train/VIInput"
labelSax_dir = "./data/Train/ASLabel"
# model directory
VImodel_dir = './model/VImodels'
ASmodel_dir = './model/ASmodels'
os.makedirs(ASmodel_dir, exist_ok=True)
# log directory
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)

# create a logger
logger = logging.getLogger('mylogger')
# set logger level
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(log_dir + '/mylog.txt')
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# tensorboard
writer = SummaryWriter("AS")

# -------------------------------------#
# -------------Load data---------------#
# -------------------------------------#
# load data
train_data, train_label, valid_data, valid_label = get_data(inputSax_dir, labelSax_dir, stage=2)
# normalization
train_data = normalization(train_data)
train_label = normalization(train_label)
valid_data = normalization(valid_data)
valid_label = normalization(valid_label)

# -------------------------------------#
# -------------Define paras------------#
# -------------------------------------#

# define loss function
loss_MSE = nn.MSELoss()
loss_VGG = VGGLoss()

# define the number of epochs
start_epoch = 0
end_epoch = 700
# learning rate
lr = 1e-4
# batch size
batch_size = 1

# -------------------------------------#
# -------------Define network----------#
# -------------------------------------#

# VI stage
netVI = VINet()
netVI = netVI.to(device)
# load existing VI network
netVI.load_state_dict(torch.load('%s/netVI_params_%d.pkl' % (VImodel_dir, 700)))
netVI.eval()

# AS stage
netAS = ASNet()
netAS = netAS.to(device)
if start_epoch > 0:
    netAS.load_state_dict(torch.load('%s/netAS_params_%d.pkl' % (ASmodel_dir, start_epoch)))

# optimizer
optimizerAS = torch.optim.Adam(params=netAS.parameters(), lr=lr)

# -------------------------------------#
# -------------Start training----------#
# -------------------------------------#

start = time.time()
numIter = 0  # to record the number of iterations
for epoch in range(start_epoch + 1, end_epoch + 1):

    # Training process
    dataset = TraindataSet(train_data, train_label, transform=randomTransform)
    train_iter = DataLoader(dataset, batch_size, num_workers=0, shuffle=True)

    netAS.train()
    i = 0  # to record the number of batch
    for x, y in train_iter:

        i = i + 1
        numIter = numIter + 1
        torch.cuda.empty_cache()

        VIinput = x.cuda()
        ASlabel = y.cuda()

        # VI stage
        VIoutput = netVI(VIinput)
        # AS stage
        ASinput = torch.cat((VIinput, VIoutput), 1)
        ASoutput = netAS(ASinput)

        # loss
        train_loss_MSE = loss_MSE(ASoutput, ASlabel)
        train_loss_VGG = loss_VGG(ASoutput, ASlabel)
        loss = 1 * train_loss_MSE + 0.01 * train_loss_VGG

        # update AS network
        optimizerAS.zero_grad()
        loss.backward()
        optimizerAS.step()

        # print loss
        print(
            "[Epoch %d/%d] [Batch %d/%d] [MSE: %f] [VGG: %f]"
            % (epoch, end_epoch, i, len(train_iter), train_loss_MSE.item(), train_loss_VGG.item()))
        logger.info("[Epoch %d/%d] [Batch %d/%d] [MSE: %f] [VGG: %f]"
                    % (epoch, end_epoch, i, len(train_iter), train_loss_MSE.item(), train_loss_VGG.item()))

        writer.add_scalars('Loss/TrainMSE', {'MSE': train_loss_MSE.item()}, numIter)
        writer.add_scalars('Loss/TrainVGG', {'VGG': train_loss_VGG.item()}, numIter)

        # to record images
        VIinput2d = torch.unsqueeze(torch.squeeze(VIinput), 1)
        VIoutput2d = torch.unsqueeze(torch.squeeze(VIoutput), 1)
        ASoutput2d = torch.unsqueeze(torch.squeeze(ASoutput), 1)
        ASlabel2d = torch.unsqueeze(torch.squeeze(ASlabel), 1)
        if numIter % 25 == 1:
            writer.add_images('TrainImages/VIinput', VIinput2d, int((numIter - 1) / 25), dataformats='NCHW')
            writer.add_images('TrainImages/VIoutput', VIoutput2d, int((numIter - 1) / 25), dataformats='NCHW')
            writer.add_images('TrainImages/ASoutput', ASoutput2d, int((numIter - 1) / 25), dataformats='NCHW')
            writer.add_images('TrainImages/ASlabel', ASlabel2d, int((numIter - 1) / 25), dataformats='NCHW')

    # Valid process
    netAS.eval()
    if valid_label is not None:

        valid_dataset = TraindataSet(valid_data, valid_label)
        valid_iter = DataLoader(valid_dataset, batch_size)
        valid_loss_MSE, valid_loss_VGG = 0, 0

        for x, y in valid_iter:
            with torch.no_grad():

                VIinput = x.cuda()
                ASlabel = y.cuda()

                # VI stage
                VIoutput = netVI(VIinput)
                # AS stage
                ASinput = torch.cat((VIinput, VIoutput), 1)
                ASoutput = netAS(ASinput)

                # loss
                valid_loss_MSE = valid_loss_MSE + loss_MSE(ASoutput, ASlabel).item()
                valid_loss_VGG = valid_loss_VGG + loss_VGG(ASoutput, ASlabel).item()

                # to record images
                VIinput2d = torch.unsqueeze(torch.squeeze(VIinput), 1)
                VIoutput2d = torch.unsqueeze(torch.squeeze(VIoutput), 1)
                ASoutput2d = torch.unsqueeze(torch.squeeze(ASoutput), 1)
                ASlabel2d = torch.unsqueeze(torch.squeeze(ASlabel), 1)
                if epoch % 25 == 1:
                    writer.add_images('ValidImages/VIinput', VIinput2d, int(epoch), dataformats='NCHW')
                    writer.add_images('ValidImages/VIoutput', VIoutput2d, int(epoch), dataformats='NCHW')
                    writer.add_images('ValidImages/ASoutput', ASoutput2d, int(epoch), dataformats='NCHW')
                    writer.add_images('ValidImages/ASlabel', ASlabel2d, int(epoch), dataformats='NCHW')

        writer.add_scalars('Loss/ValidMSE', {'MSE': valid_loss_MSE / len(valid_iter)}, epoch)
        writer.add_scalars('Loss/ValidVGG', {'VGG': valid_loss_MSE / len(valid_iter)}, epoch)

    # save AS models
    if epoch % 50 == 0:
        torch.save(netAS.state_dict(), "%s/netAS_params_%d.pkl" % (ASmodel_dir, epoch))

# time summary
train_duration = time.time() - start
print('Training complete in {:.0f}m {:.0f}s'.format(train_duration // 60, train_duration % 60))
logger.info('Training complete in {:.0f}m {:.0f}s'.format(train_duration // 60, train_duration % 60))

writer.close()
