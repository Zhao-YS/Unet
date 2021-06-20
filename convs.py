import torch
from torch import nn

#Three type of convs


#Get the image input
class First_conv(nn.Module):
	def __init__(self,in_channel):
		super().__init__()
		self.first_conv = nn.Sequential(
			nn.Conv2d(in_channel,64,kernel_size = 3,padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64,64,kernel_size=3,padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU()
			)
	def forward(self,x):
		y = self.first_conv(x)
		return y


#The conv in the middle of the Unet(Have upward type and downward type)
class Big_conv(nn.Module):
	def __init__(self,in_channel,out_channel,mid_channel=None):
		super().__init__()
		if not mid_channel:
			mid_channel = out_channel
		self.big_conv = nn.Sequential(
			nn.Conv2d(in_channel,mid_channel,kernel_size=3,padding=1),
			nn.BatchNorm2d(mid_channel),
			nn.ReLU(),
			nn.Conv2d(mid_channel,out_channel,kernel_size=3,padding=1),
			nn.BatchNorm2d(out_channel),
			nn.ReLU()
			)
	def forward(self,x):
		y = self.big_conv(x)
		return y

#Get the image output
class Final_conv(nn.Module):
	def __init__(self,in_channel,out_channel):
		super(Final_conv,self).__init__()
		self.conv = nn.Sequential(
				nn.Conv2d(in_channel,out_channel,kernel_size=1),
				nn.BatchNorm2d(out_channel),
				nn.Sigmoid()
				)
	def forward(self,x):
		y = self.conv(x)
		return y