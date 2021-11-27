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


def yolov4(image, LABELS, draw_image, _decode):

    image, boxes, scores, classes = _decode.detect_image(image, draw_image)

    number = len(classes)

    ID = []
    j = 0
    for j in range(number):
        id = classes[j]
        ID.append(LABELS[id])

    xmin = []
    i = 0
    for i in range(number):
        xmin.append(boxes[i][0])

    return ID, scores, xmin


def recognize_overlapping_bbox_algorithm(ID, scores, xmin, unusual_distance):
    sorted_xmin, sorted_id = (list(t) for t in zip(*sorted(zip(xmin, ID))))
    same_xmin, sorted_score = (list(q) for q in zip(*sorted(zip(xmin, scores))))

    dict1 = dict(zip(sorted_score, sorted_id))
    dict2 = dict(zip(sorted_score, sorted_xmin))

    i = 0
    j = 0
    tolerance = 0
    tolerance_list = []
    distance = []

    for i in range(len(sorted_xmin) - 1):
        j = i + 1
        if j == len(sorted_xmin):
            break
        else:
            dis = sorted_xmin[j] - sorted_xmin[i]
            distance.append(dis)

            if dis < unusual_distance:
                tolerance = tolerance + 1
                if tolerance_list == []:
                    tolerance_list.append(sorted_score[i])
                    tolerance_list.append(sorted_score[j])
                else:
                    tolerance_list.append(sorted_score[j])

                if tolerance >= 2:
                    max_score = max(tolerance_list)
                    # print("there are continuosly overlapped bbox!")
                    # print("tolerance_list:",tolerance_list)
                    # print("max_score:",max_score)
                    for element in tolerance_list:
                        if element < max_score:
                            if element in dict1:
                                dict1.pop(element)
                else:
                    # print("there are overlapped bbox!")
                    if sorted_score[i] > sorted_score[j]:
                        if sorted_score[j] in dict1:
                            dict1.pop(sorted_score[j])
                            dict2.pop(sorted_score[j])  ###########################################
                    else:
                        if sorted_score[i] in dict1:
                            dict1.pop(sorted_score[i])
                            dict2.pop(sorted_score[j])  #############################
            else:
                tolerance = 0
                tolerance_list.clear()
                continue

    final_id = []
    final_score = []
    add = 0

    for key, value in dict1.items():
        final_id.append(value)
        final_score.append(key)
        add = add + key

    final_xmin = []

    for key, value in dict2.items():
        final_xmin.append(value)

    return final_id, final_score, final_xmin


def adjustment_flip_algorithm(weight_path, input_ID_A, input_ID_B, input_score_A, input_score_B):
    file = open(weight_path, 'r')

    weight_dict = {}
    for line in file.readlines():
        line = line.strip()
        k = line.split(' ')[0]
        v = line.split(' ')[1]
        weight_dict[k] = int(v)

    file.close()

    new_score_A_list = []
    for i in input_ID_A:
        # print(f"key: {i}, value:",weight_dict[i])
        if weight_dict[i] == 1:
            lista_location = input_ID_A.index(i)
            # print("catch! Location is in:",lista_location)
            new_score_A_list.append(input_score_A[lista_location])

    if len(new_score_A_list):
        #print("new_a:", new_score_A_list)
        add = 0
        for j in new_score_A_list:
            add = add + j

        # print("add_a:",add)
        average_score_a = add / len(new_score_A_list)
        # print("new_score_a:",average_score_a)
    else:
        add = 0
        for j in input_score_A:
            add = add + j

        # print("add_a:",add)
        average_score_a = add / len(input_score_A)

    new_score_B_list = []
    for k in input_ID_B:
        # print(f"key: {k}, value:",weight_dict[k])
        if weight_dict[k] == 1:
            listb_location = input_ID_B.index(k)
            # print("catch! Location is in:",listb_location)
            new_score_B_list.append(input_score_B[listb_location])

    if len(new_score_B_list):
        #print("new_b:", new_score_B_list)
        add = 0
        for l in new_score_B_list:
            add = add + l

        average_score_b = add / len(new_score_B_list)
        #print("new_score_b:", average_score_b)

    else:
        add = 0
        for l in input_score_B:
            add = add + l

        # print("add_a:",add)
        average_score_b = add / len(input_score_B)

    return average_score_a, average_score_b