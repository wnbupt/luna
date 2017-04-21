from skimage import io,data,color
from skimage.measure import label,regionprops, perimeter
from numpy import *
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
from skimage.morphology import ball, disk,binary_opening, dilation,binary_dilation,binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
import numpy as np
import cv2
#这里放入要读取的图片
filename = 'C:\\Users\\Administrator\\Desktop\\边缘检测\\3.jpg'
img = io.imread(filename)
#转为灰度图
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
img = gray
#二值化
a,binary = cv2.threshold(gray,130,255,cv2.THRESH_BINARY)
binary = 255 - binary;
#以下步骤是填充孔洞
#contours存放着边缘
a,contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
m,n = binary.shape;
a = zeros((m,n));
a = np.uint8(a);
#a里保存着边缘轮廓信息
cv2.drawContours(a,contours,-1,(255,0,0),1) 
#填充孔洞
binary = ndi.binary_fill_holes(a)
lenA,lenB = binary.shape;
#a中保存着填充孔洞完后的二值化的图像
a = zeros((lenA,lenB))
for i in range(lenA):
    for j in range(lenB):
        if binary[i][j] == True:
            a[i][j] = 1;

#调用skimage的clear_board，清除边缘
a = clear_border(a)
#定义大小为2的结构圆，先经行腐蚀，防止两个肺中间的那个东西和肺连在一起
se = disk(2);
a = binary_erosion(a,se);
a = label(a)
plt.imshow(a);
plt.show();
#!!这里假设肺只有两个，去连通区域面积最大的两块
areas = [r.area for r in regionprops(a)]
areas.sort()
if len(areas) > 2:
    for region in regionprops(a):
        if region.area < areas[-2]:
             for coordinates in region.coords:
                   a[coordinates[0], coordinates[1]] = 0

#对原图像进行膨胀，还原回去
a = binary_dilation(a,se);
plt.imshow(a);
plt.show();


for i in range(lenA):
    for j in range(lenB):
        if a[i][j] > 0:
            a[i][j] = 1;
a = np.uint8(a);
#分开两个区域做滚球法
a = label(a)
a2 = a == 1;
a1 = a == 2;
#定义大小为10的结构体
se = disk(10);
#分别对两个体做形态学开操作
a1 = binary_closing(a1,se);
a2 = binary_closing(a2,se);
a = a1 + a2;

#与原始的图像做乘运算，截取目标区域
result = a * img;
plt.imshow(result);
plt.show();

