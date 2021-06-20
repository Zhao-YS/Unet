import numpy as np
import cv2
from imagecodecs import *
import os
from skimage import transform

Path1 = 'raw/train_img/'
Path2 = 'raw/train_label/'
Path1a = 'Train_images/'
Path2a = 'Train_labels/'

Path_tlabel = 'test_label/'
Path_Ntlabel = 'New_test_label/'
Path_timg = 'test_img/'
Path_Ntimg = 'New_test_img/'

def main():
  #train data Augmentation
  for root, dirs, files in os.walk(Path1):
    for name in files:
        img = cv2.imread(Path1+name)
        #Image itself
        np.save(Path1a+'pic'+name[0:-4]+'-0'+'.png.npy',img.astype(np.uint8))
        #horizontally flip
        img1 = np.fliplr(img)
        np.save(Path1a+'pic'+name[0:-4]+'-1'+'.png.npy',img1.astype(np.uint8))
        #Vertical filp
        img1 = np.flipud(img)
        np.save(Path1a+'pic'+name[0:-4]+'-2'+'.png.npy',img1.astype(np.uint8))
        #rotate 90
        img1 = transform.rotate(img,90,preserve_range=True).astype(np.uint8)
        np.save(Path1a+'pic'+name[0:-4]+'-3'+'.png.npy',img1.astype(np.uint8))
        #rotate 180
        img1 = transform.rotate(img,180,preserve_range=True).astype(np.uint8)
        np.save(Path1a+'pic'+name[0:-4]+'-4'+'.png.npy',img1.astype(np.uint8))
        #rotate 270
        img1 = transform.rotate(img,270,preserve_range=True).astype(np.uint8)
        np.save(Path1a+'pic'+name[0:-4]+'-5'+'.png.npy',img1.astype(np.uint8))


  for root, dirs, files in os.walk(Path2):
    #test data Augmentation
      for name in files:
        if 'npy' in name:
          img = np.load(Path2+name)
          img = 2-img
          np.save(Path2a+'pic'+name[0:-8]+'-0'+'.png.npy',img.astype(np.uint8))
          img1 = np.fliplr(img)
          np.save(Path2a+'pic'+name[0:-8]+'-1'+'.png.npy',img1.astype(np.uint8))
          img1 = np.flipud(img)
          np.save(Path2a+'pic'+name[0:-8]+'-2'+'.png.npy',img1.astype(np.uint8))
          img1 = transform.rotate(img,90,preserve_range=True).astype(np.uint8)
          np.save(Path2a+'pic'+name[0:-8]+'-3'+'.png.npy',img1.astype(np.uint8))
          img1 = transform.rotate(img,180,preserve_range=True).astype(np.uint8)
          np.save(Path2a+'pic'+name[0:-8]+'-4'+'.png.npy',img1.astype(np.uint8))
          img1 = transform.rotate(img,270,preserve_range=True).astype(np.uint8)
          np.save(Path2a+'pic'+name[0:-8]+'-5'+'.png.npy',img1.astype(np.uint8))

    
    #turn the label into 0/1 matrix, And rotate the first test image 180.
  for root, dirs, files in os.walk(Path_tlabel):
    for name in files:
        img = cv2.imread(Path_tlabel+name)
        if '0' in name:
            img = transform.rotate(img,180,preserve_range=True).astype(np.uint8)
        img = 1 - np.array(img.sum(axis = 2)) / 765
        np.save(Path_Ntlabel+name+'.npy',img.astype(np.uint8))
  
  for root, dirs, files in os.walk(Path_timg):
    for name in files:
        img = cv2.imread(Path_timg+name)
        np.save(Path_Ntimg+name+'.npy',img.astype(np.uint8))
main()