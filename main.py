#! /usr/bin/env python
import time
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F

from make_Dataset import dataset_make	
from model_Train import train
from predict import Prediction
from net_Structure import UNet,Down_ward,Up_ward
from convs import First_conv, Big_conv, Final_conv



#main function, can set the parameters such as Learning rate, batch size
if __name__ == '__main__':
	batch_size = 3 
#Load the data
	train_iter, test_iter = dataset_make(batch_size)
	print('Data processed\n')	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#print(device)
#Generate the Unet
	net = nn.Sequential(UNet(3))
	for param in net.parameters():
		nn.init.normal_(param,mean=0,std=0.01)
	print('Param initialized\n')
	loss =nn.BCELoss()
#Set the optimizer
	optimizer = torch.optim.Adam(net.parameters(),lr=0.0003)
	num_epoch = 15 
	#print(net)
	#print('Train start')
#Train, load the best model, make prediction
	#train(net,train_iter,test_iter,loss,num_epoch,batch_size,optimizer,device)
	#net = torch.load('unet-15.model')
	#Prediction(test_iter,net,device,512)
	
