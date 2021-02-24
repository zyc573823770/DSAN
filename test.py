import argparse
import json
import os
import pickle
import sys
from copy import deepcopy
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import torch.utils.data as data
import torchvision.datasets as dst
import torchvision.models as models
import torchvision.models.resnet as resn
import torchvision.transforms as T
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm, trange

import mmd
from Weight import Weight

# a = torch.randn(10,5)
# b = torch.randn(10,5)
# ss, tt, st = Weight.cal_weight(F.softmax(b, dim=1), F.softmax(b, dim=1), class_num=5, comp=True)
# print(ss)
a = [1,2]
b, c = a[0], a[1]
print(b, c)