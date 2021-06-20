import time
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
import os
from cv2 import *

from convs import First_conv, Big_conv, Final_conv
from net_Structure import UNet, Down_ward, Up_ward
from model_Train import train, Evaluate


#make prediction and save the result
def Prediction(data_iter,net,device,width):
	i = 0
	k = 0
	No = [4,1,3,0,2]
	for x,y in data_iter:
		x = x.to(device)
		y = y.to(device)
		y_pre = net(x)
		y_pre = y_pre.reshape(-1,width,width)
		#print(y_pre.shape)
		#print(y.shape)
		y_pre = ((y_pre>=0.5)*1)*255
		y_pre = y_pre.cpu().numpy()
		y = y*255
		y = y.cpu().numpy()
		y_pre = np.uint8(y_pre)
		y = np.uint8(y)
		for j in range(y_pre.shape[0]):
			cv2.imwrite('Pre_label/'+'pre'+str(No[k])+'.png',255-y_pre[j])
			k = k+1
		i += 1
