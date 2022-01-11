# ReadMe file

## Environment

- python==3.7.5
- pytorch==1.71
- cv2==4.5.1

## File Description 

#### data_Preprocess.py:data augmentation 

Run Code:

```
python3 data_Preprocess.py
```

#### main.py:The main program, which controls the training, testing, and import of the model 

Run Code:

```- 
python3 main.py
```

- Train

Enter main.py, uncomment the 36th line, you can call the train function for training, the model obtained by 15 epochs is generated in the current folder 

- Import Model

Uncomment line 37 and import the model you want to predict in line 37 of main.py, here the default is unet-15 

- Predict

 Uncomment line 37 and import the model you want to predict in line 37 of main.py, here the default is unet-15 

#### result_process.py:Analyze the results and output other evaluation indicators 

Run Code:

```
 python3 result_process.py
```

#### v_info_v_rand.bsh:Calcutalte VInfo，VRand

Run Code:Execute in ImageJ 

#### convs.py:Three types of conv 

#### net_structure:Main structure of Unet network 

#### make_Dataset.py:Create training sets and test sets that meet the needs of pytorch 

#### model_train:Define the process of model training 

#### predict.py:Make predictions and output the results to Pre_label 

#### 

## Folder Description 

#### Pre_label:Make Predictions

#### raw：raw training set

#### test_img:raw test set

#### test_label: label of data in test set

### The following need to execute data_Preprocess to generate

#### New_test_img:New test image (flipped and changed to matrix form) 

#### New_test_label:New training picture (flipped and changed to matrix form) 

#### Train_images:Images after training set augmentation 

#### Train_labels:Labels after training set augmentation 

