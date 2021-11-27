from collections import deque
import datetime
import cv2
import os
import time
import numpy as np
import torch
import xml.etree.ElementTree as ET
import shutil
import glob
from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from model.decode_np import Decode
import logging
from yolov4_module import *

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
# 6G的卡，训练时如果要预测，则设置use_gpu = False，否则显存不足。
use_gpu = True

# 調參區 ############################################################################################################

path_name = r"D:\OCR\id\final_test\test_img"  # 要預測圖片的位置
model_path = './weights/addsize_model.pt'
classes_path = './data/new_classes.txt'
weight_path = r"C:\Users\jason\Pytorch-YOLOv4-master\adjustment_flip_algorithm_weight.txt"
test_number = 500  # 指定跑幾張test
unusual_distance = 10

LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
          "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# input_shape越大，精度会上升，但速度会下降。
input_shape = (608, 608)               # input_shape = (320, 320), input_shape = (416, 416)

# 验证时的分数阈值和nms_iou阈值
conf_thresh = 0.05
nms_thresh = 0.45

# 是否给图片画框。不画可以提速。读图片、后处理还可以继续优化。
draw_image = False

num_anchors = 3
all_classes = get_classes(classes_path)
num_classes = len(all_classes)

# 前置準備 ########################################################################################################
print(" ")
print(f"loading yolov4 model:'{model_path}'")
yolo = YOLOv4(num_classes, num_anchors)
if torch.cuda.is_available():  # 如果有gpu可用，模型（包括了权重weight）存放在gpu显存里
    yolo = yolo.cuda()
yolo.load_state_dict(torch.load(model_path))
yolo.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式. 不这样做的化会产生不一致的推理结果.

_decode = Decode(conf_thresh, nms_thresh, input_shape, yolo, all_classes)

if not os.path.exists('./images/res/'): os.mkdir('./images/res/')

path_dir = os.listdir(path_name)

file = open(weight_path, 'r')

print(f"loading flip weight:'{weight_path}'")
# 遍历文本文件的每一行，strip可以移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
weight_dict = {}
for line in file.readlines():
    line = line.strip()
    k = line.split(' ')[0]
    v = line.split(' ')[1]
    weight_dict[k] = int(v)

file.close()
print(weight_dict)
###########################################################################################################

time_stat = deque(maxlen=20)
start_time = time.time()
end_time = time.time()
num_imgs = len(path_dir)
start = time.time()
print(" ")
print("start detecting!")
print(" ")

# 開始跑整個資料夾

num = 0
for k, filename in enumerate(path_dir[:test_number]):

    num = num + 1
    image = cv2.imread(path_name + '/' + filename)
    img_flip_along_xy = cv2.flip(image, -1)

    a1 = yolov4(image, LABELS, draw_image, _decode)
    # print("a1:",a1[0])
    a2 = recognize_overlapping_bbox_algorithm(a1[0], a1[1], a1[2], unusual_distance)

    b1 = yolov4(img_flip_along_xy, LABELS, draw_image, _decode)
    # print("b1:", b1[0])
    b2 = recognize_overlapping_bbox_algorithm(b1[0], b1[1], b1[2], unusual_distance)

    # 判斷flip區
    c = adjustment_flip_algorithm(weight_path, a2[0], b2[0], a2[1], b2[1])


    if c[0] > c[1]:
        listc = a2[0]
        listc_wrong = b2[0]
    elif c[0] == c[1]:
        lictc = listc_wrong =["e", "r", "r", "o", "r"]
        print("同分，比較不出!")
    else:
        listc = b2[0]
        listc_wrong = a2[0]

    print(f"img_{num}:", filename)
    print("final ID:", listc)
    print("ID_a:", a2[0])
    print("ID_b:", b2[0])
    print("score_a:",a2[1])
    print("score_b:", b2[1])

#######################################################

    if len(a2[0]) == len(b2[0]):
        reverse = b2[1][::-1]
        a_array = np.array(a2[1])
        b_array = np.array(reverse)
        diff = a_array - b_array
        print("diff:", diff)
        print(" ")
        k=0
        for i in diff:
            if i >0:
                aa=a2[0][k]
                inform = f"{aa},p"
                fileObject = open('sampleList.txt', 'a')
                fileObject.write(inform)
                fileObject.write('\n')
                fileObject.close()
            elif i <0:
                aa = a2[0][k]
                inform = f"{aa},n"
                fileObject = open('sampleList.txt', 'a')
                fileObject.write(inform)
                fileObject.write('\n')
                fileObject.close()
            k=k+1



########################################################


# 估计剩余时间
start_time = end_time
end_time = time.time()
time_stat.append(end_time - start_time)
time_cost = np.mean(time_stat)
eta_sec = (num_imgs - k) * time_cost
eta = str(datetime.timedelta(seconds=int(eta_sec)))

# logger.info('Infer iter {}, num_imgs={}, eta={}.'.format(k, num_imgs, eta))
# if draw_image:
#     cv2.imwrite('./images/res/' + filename, image)
#     logger.info("Detection bbox results save in images/res/{}".format(filename))

cost = time.time() - start
logger.info('total time: {0:.6f}s'.format(cost))
logger.info('Speed: %.6fs per image,  %.1f FPS.' % ((cost / num_imgs), (num_imgs / cost)))

