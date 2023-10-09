# import libraries
import os

import time
import random
import torch
import numpy as np

from utils import normalization
from network import VINet, ASNet
from scipy.io import loadmat
import scipy.io as sio

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
# model directory
VImodel_dir = './model/VImodels'
ASmodel_dir = './model/ASmodels'

# -------------------------------------#
# ----------Testing process------------#
# -------------------------------------#

# testing process
stage = 2  # 1 for VI stage, 2 for AS stage
# choose model
epoch_num = 700
# VI stage
netVI = VINet()
netVI = netVI.to(device)
netVI.load_state_dict(torch.load('%s/netVI_params_%d.pkl' % (VImodel_dir, epoch_num)))
netVI.eval()
# AS stage
if stage == 2:
    netAS = ASNet()
    netAS = netAS.to(device)
    netAS.load_state_dict(torch.load('%s/netAS_params_%d.pkl' % (ASmodel_dir, epoch_num)))
    netAS.eval()

# output directory
outputDir = './TestOutput'
os.makedirs(outputDir, exist_ok=True)


# testing kernel
def TestKernel(VIinput, stage=2):

    with torch.no_grad():
        start = time.time()
        VIinput = VIinput.cuda()
        VIoutput = netVI(VIinput)
        if stage == 2:
            ASinput = torch.cat((VIinput, VIoutput), 1)
            ASoutput = netAS(ASinput)

        duration = time.time() - start
        print('Testing complete in {:.0f}m {:.2f}s'.format(duration // 60, duration % 60))

        if stage == 2:
            return VIoutput, ASoutput
        elif stage == 1:
            return VIoutput


# load testing data
dataDir = './data/Test'
crrDataName = dataDir + "/CropSax" + str(1) + ".mat"
crrData = loadmat(crrDataName)
crrData = torch.FloatTensor(crrData['CropSax'])


for idxSlice in range(crrData.size(0)):
    crrCine = torch.unsqueeze(torch.unsqueeze(crrData[idxSlice], 0), 0)  # single slice
    crrCine = normalization(crrCine)  # normalization
    if stage == 1:
        VIoutput = TestKernel(VIinput=crrCine, stage=1)
    else:
        VIoutput, ASoutput = TestKernel(VIinput=crrCine)

    arrayOutput = np.array(torch.squeeze(VIoutput).cpu()).astype(np.float64)
    matName = "VIoutput" + str(idxSlice)
    mdic = {matName: arrayOutput}
    sio.savemat(os.path.join(outputDir, matName + '.mat'), mdic)
    if stage == 2:
        arrayOutput = np.array(torch.squeeze(ASoutput).cpu()).astype(np.float64)
        matName = "ASoutput" + str(idxSlice)
        mdic = {matName: arrayOutput}
        sio.savemat(os.path.join(outputDir, matName + '.mat'), mdic)


