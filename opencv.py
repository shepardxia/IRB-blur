import cv2
import numpy as np
import matplotlib.pyplot as plt
 
#1、imread加载图片
img = cv2.imread('./images/07.jpg')
 
#2、将图像转换为灰度图
 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#2、高斯平滑模糊
#GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY=0, int borderType=BORDER_DEFAULT )
#Size ksize必须为正奇数
img = cv2.GaussianBlur(img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
 
#3、中值滤波(池化)，消除噪音数据，medianBlur(InputArray src, OutputArray dst, int ksize)   ksize必须为奇数
img = cv2.medianBlur(img, 5)
 
 
#4、利用Sobel方法可以进行sobel边缘检测，突出边缘
img = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
 
#图像的二值化就是将图像上的像素点的灰度值设置为0或255，这样将使整个图像呈现出明显的黑白效果，<150的全为黑，>150的全为白
ret, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
 
#膨胀，让轮廓突出
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
img = cv2.dilate(binary, element1, iterations=1)
#腐蚀
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
img = cv2.erode(img, element2, iterations=1)
#膨胀，让轮廓更明显
img = cv2.dilate(img, element1, iterations=3)

cv2.imwrite('precossed.jpg', img)
 
###############################################
# 查找轮廓(img: 原始图像，contours：矩形坐标点，hierarchy：图像层次)
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
max_ratio = -1
ratios = []
num = 0


print(len(contours))
for i in range(len(contours)):
    cnt = contours[i]
 
    #计算轮廓面积
    area = cv2.contourArea(cnt)
    if area < 100:
        continue
 
    #四边形的最小外接矩形,得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    rect = cv2.minAreaRect(cnt)
 
    # 矩形的四个坐标（顺序不定，但是一定是一个左下角、左上角、右上角、右下角这种循环顺序(开始是哪个点未知)）
    box = cv2.boxPoints(rect)
    # 转换为long类型
    box = np.int0(box)
 
    # 计算长宽高
    height = abs(box[0][1] - box[2][1])
    weight = abs(box[0][0] - box[2][0])
    ratio = float(weight) / float(height)
    # 正常的车牌宽高比在2.7~5之间
    if ratio > max_ratio:
        max_box = box
 
    #if ratio > 5.5 or ratio < 2:
    #    continue
 
    num +=1
    ratios.append((max_box,ratio))
 

#返回就是车牌的矩阵的四个点的坐标
box = ratios[0][0]
print(box)
print(box[0,1])
 
ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
print(ys)
xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
 
ys_sorted_index = np.argsort(ys)
print(ys_sorted_index)
xs_sorted_index = np.argsort(xs)
 
# 获取x上的坐标
x1 = box[xs_sorted_index[0], 0]
print(x1)
x2 = box[xs_sorted_index[3], 0]
print(x2)
 
# 获取y上的坐标
y1 = box[ys_sorted_index[0], 1]
print(y1)
y2 = box[ys_sorted_index[3], 1]
#
img2 = cv2.imread('./images/07.jpg')
# # 截取图像
img_plate = img2[y1:y2, x1:x2]
cv2.imwrite('./test1.jpg', img_plate)