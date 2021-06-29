# ReadMe file

## 开发环境

- python==3.7.5
- pytorch==1.71
- cv2==4.5.1

## 文件说明

#### data_Preprocess.py:进行数据的扩增

执行方式:

```
python3 data_Preprocess.py
```

#### main.py:主程序,控制模型的训练，测试，与导入

执行方式:

```- 
python3 main.py
```

- 训练

进入main.py,取消第36行的注释,即可调用train函数进行训练，15个epoch得到的模型在当前文件夹下生成

- 导入模型

  取消对37行的注释，在main.py的37行中导入进行想要预测的模型，这里默认是unet-15

- 预测

  取消对38行的注释，预测的图片输入到Pre_label中

#### result_process.py:进行结果的分析，输出其他评价指标

执行方式:

```
 python3 result_process.py
```

#### v_info_v_rand.bsh:计算VInfo，VRand

执行方式:在ImageJ中执行

#### convs.py:三类conv

#### net_structure:Unet网络主要结构

#### make_Dataset.py:制作符合pytorch需要的训练集与测试集

#### model_train:定义模型训练的过程

#### predict.py:进行预测，输出结果到Pre_label中

#### 

## 文件夹说明

#### Pre_label:预测结果

#### raw：训练集原始数据

#### test_img:测试集原始图片

#### test_label:测试集原始label

### 以下需要执行data_Preprocess才会生成

#### New_test_img:新的测试图片(进行了翻转,且改为矩阵的形式)

#### New_test_label:新的训练图片(进行了翻转,且改为矩阵的形式)

#### Train_images:训练集扩增后图片

#### Train_labels:训练集扩增后标签
