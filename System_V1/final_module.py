from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from model.decode_np import Decode

import torch

def yolov4(image, model_path, classes_path):

    # input_shape越大，精度会上升，但速度会下降。
    input_shape = (608, 608)  # input_shape = (320, 320), input_shape = (416, 416)

    # 验证时的分数阈值和nms_iou阈值
    conf_thresh = 0.05
    nms_thresh = 0.45

    num_anchors = 3
    all_classes = get_classes(classes_path)
    num_classes = len(all_classes)
    yolo = YOLOv4(num_classes, num_anchors)
    if torch.cuda.is_available():  # 如果有gpu可用，模型（包括了权重weight）存放在gpu显存里
        yolo = yolo.cuda()

    yolo.load_state_dict(torch.load(model_path))
    yolo.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式. 不这样做的化会产生不一致的推理结果.

    _decode = Decode(conf_thresh, nms_thresh, input_shape, yolo, all_classes)

    draw_image = False
    image, boxes, scores, classes = _decode.detect_image(image, draw_image)

    return boxes, scores, classes


def recognize_overlapping_bbox_algorithm(boxes, scores, classes, LABELS, unusual_distance):

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


def flip_algorithm(input_ID_A, input_ID_B, input_score_A, input_score_B):

    add=0
    for i in input_score_A:
        add=add+i

    average_score_a = add / len(input_score_A)

    add = 0
    for j in input_score_B:
        add = add + j

    average_score_b = add / len(input_score_B)

    if average_score_a > average_score_b:
        predict = input_ID_A
    elif average_score_a == average_score_b:
        predict = "同分無法比較!"
    else:
        predict = input_ID_B

    return predict