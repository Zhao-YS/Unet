import time
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
import os



#Making training and test set
def dataset_make(batch_size):
	images_list = os.listdir('./Train_images/')
	test_list = os.listdir('./New_test_img/')
	images = []
	labels = []
	test_images = []
	test_labels = []
	for i in images_list:
		image_dir = './Train_images/' + i
		labels_dir = './Train_labels/' + i
		images.append(np.load(image_dir))
		labels.append(np.load(labels_dir))
	for i in test_list:
		image_dir = './New_test_img/' + i
		labels_dir = './New_test_label/' + i
		test_images.append(np.load(image_dir))
		test_labels.append(np.load(labels_dir))
	images = torch.tensor(np.array(images).astype("float32"))
	labels = torch.tensor(np.array(labels).astype("float32"))
	test_images = torch.tensor(np.array(test_images).astype("float32"))
	test_labels = torch.tensor(np.array(test_labels).astype("float32"))
#Shuffle the order of dim of matrix
	features = images.permute(0,3,1,2)
	test_features = test_images.permute(0,3,1,2)
	train_features = features
	train_labels = labels
	test_features = test_features
	test_labels = test_labels
	
	train_dataset = Data.TensorDataset(train_features,train_labels)
	test_dataset = Data.TensorDataset(test_features,test_labels)
	train_iter = Data.DataLoader(train_dataset,batch_size)
	test_iter = Data.DataLoader(test_dataset,batch_size)

	#print(images.shape)
	#print(labels.shape)	
	return train_iter, test_iter	