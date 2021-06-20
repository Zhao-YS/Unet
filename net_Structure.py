import time
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
	
from convs import First_conv, Big_conv, Final_conv



#Down-ward sampling without midchannel
class Down_ward(nn.Module):
	def __init__(self,in_channel,out_channel):
		super().__init__()
		self.downward = nn.Sequential(
			nn.MaxPool2d(2),
			Big_conv(in_channel,out_channel)
			)
	def forward(self,x):
		y = self.downward(x)
		return y

#Up-ward sampling with midchannel
class Up_ward(nn.Module):
	def __init__(self,in_channel,out_channel):
		super(Up_ward,self).__init__()
		self.upward = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
		self.conv = Big_conv(in_channel,out_channel,in_channel//2)
		
	def forward(self,x1,x2):
		x1 = self.upward(x1)
		x_change = x2.shape[2] - x1.shape[2]
		y_change = x2.shape[3] - x1.shape[3]

		x1 = F.pad(x1,[x_change//2,x_change-y_change//2,y_change//2,y_change-x_change//2])

		x = torch.cat([x2,x1],dim=1)
		y = self.conv(x)
		return y


#The main structure of UNet
class UNet(nn.Module):
	def __init__(self,in_channel):
		super(UNet,self).__init__()
		self.firstconv = First_conv(3)
		self.downward1 = Down_ward(64,128)
		self.downward2 = Down_ward(128,256)
		self.downward3 = Down_ward(256,512)
		self.downward4 = Down_ward(512,512)
		self.upward1 = Up_ward(1024,256)
		self.upward2 = Up_ward(512,128)
		self.upward3 = Up_ward(256,64)
		self.upward4 = Up_ward(128,64)
		self.out = Final_conv(64,1)



	def forward(self,x):
		first = self.firstconv(x)
		down1 = self.downward1(first)
		down2 = self.downward2(down1)
		down3 = self.downward3(down2)
		down4 = self.downward4(down3)
		y = self.upward1(down4,down3)
		y = self.upward2(y,down2)
		y = self.upward3(y,down1)
		y = self.upward4(y,first)
		final_out = self.out(y)	
		
		return final_out