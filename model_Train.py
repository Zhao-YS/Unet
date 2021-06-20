import time
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
import os
	
from convs import First_conv, Big_conv, Final_conv
from net_Structure import UNet, Down_ward, Up_ward

#Compare the prediction with the test label, get TN, TP, FN, FP
def Evaluate(data_iter, net, device):
	total_size,TPs,TNs,All_positives,All_negatives,right_pres = 0.0, 0.0, 0.0, 0.0, 0.0,0.0 
	for x,y in data_iter:
		net.eval()
		x = x.to(device)#features
		y = y.to(device)#labels
		y_pre = net(x)#prediction from features
		y_pre = y_pre.reshape(-1,512,512)
		y_pre = ((y_pre>=0.5)*1)
		Size = 512*512*3 #TP+FP+TN+FN
		All_positive = (y==1).sum().item()	#TP+FN
		TP = ((y_pre==1)*(y==1)).sum().item() 	#TP			
		tmp = (y_pre - 1)*(y - 1)
		TN = (tmp==1).sum().item()	#TN
		All_negative = Size - All_positive #TN+FP
		right_pre = (y_pre== y).sum().item() #TP+TN
		TPs += TP
		TNs += TN
		All_positives += All_positive
		All_negatives += All_negative
		total_size += Size
		right_pres += right_pre
		#print('test real_szie %f'%(Size))
		#print('test right_pre %f'%(right_pre))
		#print('test TN %f'%(TN))

		net.train()	
	return right_pres/total_size, TPs/All_positives, TNs/All_negatives

#Train the model with Adam optimizer
def train(net,train_iter,test_iter,loss,num_epoch,batch_size,optimizer,device):
	net.to(device)
	loss.to(device)
	optimizer.zero_grad()
	for epoch in range(num_epoch):
		total_size, TPs,TNs,All_positives,All_negatives,batch_count,right_pres = 0.0,0.0,0.0,0.0,0.0,0,0.0
		train_l_sum = 0.0
		for x,y in train_iter:
			start_time = time.time()
			start_time=time.time()
			x = x.to(device)
			y = y.to(device)
			#y = y.reshape(-1,512,512)
			#print(y.size())
			y_pre = net(x)
			#print(y_pre.size())
			y_pre = y_pre.reshape(-1,512,512)
			l = loss(y_pre, y).sum()
			l.backward()
			optimizer.step()
			optimizer.zero_grad()
			train_l_sum += l.item()
			y_pre = ((y_pre>=0.5)*1)
			Size = 512*512*3 #TP+FP+TN+FN
			All_positive = (y==1).sum().item() 	#TP+FN
			TP = ((y_pre==1)*(y==1)).sum().item()#TP
			tmp = (y_pre - 1)*(y - 1)
			TN = (tmp==1).sum().item()#TN
			All_negative = Size - All_positive #TN+FP
			right_pre = (y_pre==y).sum().item()#TP+TN
			TPs += TP
			TNs += TN
			All_positives += All_positive
			All_negatives += All_negative
			total_size += Size
			batch_count += 1
			right_pres += right_pre
		test_acc, test_se, test_sp = Evaluate(test_iter, net, device)	
		print('epoch %d  loss %.4f  time cost %.2f s'%(epoch+1,train_l_sum/batch_count,time.time()-start_time))
		print('train_acc %.3f  train_se %.3f  train_sp %.3f'%(right_pres/total_size,TPs/All_positives,TNs/All_negatives))
		print('test_acc %.3f  test_se %.3f  test_sp %.3f'%(test_acc, test_se, test_sp))
		torch.save(net,'unet-'+str(epoch+1)+'.model')	