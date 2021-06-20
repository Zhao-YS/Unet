import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy.lib.shape_base import _take_along_axis_dispatcher
from sklearn.metrics import roc_curve, auc
from PIL import Image

DATA_NUM= 5

def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
    #im.show()  
    width,height = im.size
    im = im.convert("L") 
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height*width,1))
    return new_data



def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))




def get_score():
    test_path = "./test_label/"
    pre_path = "./Pre_label/pre"
    res_ = []
    pre_ = []
    fpr_ = []
    tpr_ = []
    threshold_ = []
    roc_auc_ = []
    score_se = []
    score_sp = []
    score_mIOU = []
    score_Dice = []
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for i in range(0, DATA_NUM):
        image_name = "%s.png" % i
        pre_image = cv2.imread(str(pre_path) + str(image_name))
        test_image = cv2.imread(str(test_path) + str(image_name))
        res_.append(ImageToMatrix(str(test_path) + str(image_name)).tolist())#预测数据的真值
        pre_.append(ImageToMatrix(str(pre_path) + str(image_name)).tolist())#模型的预测值
        fpr_1, tpr_1, threshold_1 = roc_curve(res_[i], pre_[i])  ###计算真正率和假正率
        fpr_.append(fpr_1)
        tpr_.append(tpr_1)
        threshold_.append(threshold_1)
        roc_auc_.append(auc(fpr_1, tpr_1))


        image_size = pre_image.shape[0]*pre_image.shape[1]*pre_image.shape[2]

        loss_image_F = abs((test_image - pre_image)/255)
        loss_image_T = 1 - loss_image_F
        loss_image_TP = (test_image/255)*(pre_image/255)
        loss_image_FP = (1 - test_image/255)*(pre_image/255)
        loss_image_FN = (1 - test_image/255)*(1 - loss_image_FP)
        loss_image_TN = (test_image/255)*(1 - loss_image_FP)
        T = np.sum(loss_image_T)
        F = np.sum(loss_image_F)
        TP = np.sum(loss_image_TP)
        FP = np.sum(loss_image_FP)
        TN = np.sum(loss_image_TN)
        FN = np.sum(loss_image_FN)
        # 数据预处理

        mIOU = T/(2*image_size - T)
        score_mIOU.append(round(mIOU,3))
        dice = 2*TP/(2*TP + FP + FN)
        score_Dice.append(round(dice,3))
        sp = TN/(TN + FP)
        score_sp.append(round(sp,3))
        se = TP/(TP + FN)
        score_se.append(round(se,3))
   
    
    print(score_mIOU)
    print(score_Dice)
    print(score_se)
    print(score_sp)
    return score_mIOU, score_Dice, score_se, score_sp, fpr_, tpr_


def main():
    score_mIOU, score_Dice, score_se, score_sp, fpr_, tpr_ = get_score()
    width = 0.3
    name = []
    for i in range(DATA_NUM):  
        name.append('pre%s.png'%(i+1)) 
    x = [1,1.5,2,2.5,3]    
    #a = plt.bar(x, score_Dice, width = width,label = 'score_Dice', fc = 'b')   
    
   
    mIOU = plt.bar(x, score_mIOU, width = width, tick_label = name,fc = 'r')   
    autolabel(mIOU)  
    plt.xlabel('dataset')
    plt.ylabel('score')
    plt.title('mIOU')
    plt.legend()
    plt.show()

    Dice = plt.bar(x, score_Dice, width = width, tick_label = name,fc = 'b')   
    autolabel(Dice)  
    plt.xlabel('dataset')
    plt.ylabel('score')
    plt.title('Dice')
    plt.legend()
    plt.show()

    se = plt.bar(x, score_se, width = width, tick_label = name,fc = 'g')   
    autolabel(se)  
    plt.xlabel('dataset')
    plt.ylabel('score')
    plt.title('SE')
    plt.legend()
    plt.show()

    sp = plt.bar(x, score_sp, width = width,  tick_label = name,fc = 'y')   
    autolabel(sp)  
    plt.xlabel('dataset')
    plt.ylabel('score')
    plt.title('SP')
    plt.legend()
    plt.show()
  

    plt.figure(figsize=(8, 5))
    line_style = ['-', '--', ':', '-.', '--']
    color_space = ['darkorange', 'red', 'green', '#800080', '#D2691E']
    for i in range(DATA_NUM):
        plt.plot(fpr_[i], tpr_[i], color=color_space[i], ###假正率为横坐标，真正率为纵坐标做曲线
            lw=2, label='pre%s.png'%i, linestyle= line_style[i]) #linestyle为线条的风格（共五种）,color为线条颜色
        #print(i)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.02, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.savefig("hyh.png",dpi=600)#保存图片，dpi设置分辨率
    plt.show()

if __name__ == '__main__':
    main()
    
