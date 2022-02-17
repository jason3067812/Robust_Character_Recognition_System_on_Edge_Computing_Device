import os
import pickle

from app.recognition import Recognition

import cv2

from matplotlib import pyplot as plt

import numpy as np


chars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H",
         "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
chars_map = {index: value for index, value in enumerate(chars)}

labels_path = "/home/stephen/Workspaces/csc-projects/datasets/end_to_end/testing_data/labels"
images_path = "/home/stephen/Workspaces/csc-projects/datasets/end_to_end/testing_data/images"


def get_image_generator():
    dirfiles = os.listdir(labels_path)

    print(len(dirfiles))

    for file in dirfiles:
        with open(os.path.join(labels_path, file), 'rb') as f:
            label = pickle.load(f)
        image = cv2.imread(os.path.join(images_path, label["image_name"]))
        boxes = label["boxes"]
        arg_sort = np.argsort(boxes[:, 0, 0])

        classes = label["text"]

        if arg_sort[0] == 0:
            yield file, image, classes


def indexes_find_classes(indexes):
    labels = []
    for i in indexes:
        i = str(i)
        labels.append(chars[i])
    return labels


def comparison_chars(gt_classes, predicted_classes):
    correct_num = 0
    char_num = 0
    success = True

    if len(predicted_classes[1]) > len(gt_classes):
        text_len = len(gt_classes)
    else:
        text_len = len(predicted_classes[1])

    for i in range(text_len):
        if gt_classes[i] == predicted_classes[1][i]:
            correct_num += 1
        if predicted_classes[0][i] is False:
            success = False
        char_num += 1

    return correct_num, char_num, success


def comparison_text(gt_classes, predicted_classes):
    gt_text = "".join(gt_classes)
    predicted_text = "".join(predicted_classes[1])

    if gt_text == predicted_text:
        return 1, 1
    else:
        return 0, 1


def __plot_result_image(number, image, boxes, classes):
    for index in range(len(boxes)):
        top_left = (boxes[index][0][0], boxes[index][0][1])
        down_left = (boxes[index][3][0], boxes[index][3][1])
        down_right = (boxes[index][2][0], boxes[index][2][1])
        cv2.rectangle(image, top_left, down_right, (255, 0, 0), 2)
        cv2.putText(
            image,
            classes[index],
            down_left,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    plt.savefig(f"results/{number}.jpg")


if __name__ == "__main__":
    recognition_service = Recognition()

    gen = get_image_generator()

    total_char_num = 0
    total_text_num = 0
    total_correct_char_num = 0
    total_correct_text_num = 0
    has_doubts_text = 0

    for file, image, classe_indexes in gen:
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)

        boxes, predicted_classes = recognition_service.recognize(image)

        correct_char_num, char_num, success = comparison_chars(classe_indexes, predicted_classes)
        correct_text_num, text_num = comparison_text(classe_indexes, predicted_classes)

        if success is True:
            if correct_text_num != 1:
                print("Error", total_text_num, file)
                __plot_result_image(correct_text_num, image, boxes, predicted_classes)
            total_char_num += char_num
            total_text_num += text_num
            total_correct_char_num += correct_char_num
            total_correct_text_num += correct_text_num
        else:
            has_doubts_text += 1
            print("Confused", total_text_num, file)
            __plot_result_image(correct_text_num, image, boxes, predicted_classes)

        if total_text_num % 100 == 0:
            print(total_char_num)
            print(total_correct_char_num)
            print(total_text_num)
            print(total_correct_text_num)
            print(has_doubts_text)

    print(total_char_num)
    print(total_correct_char_num)
    print(total_text_num)
    print(total_correct_text_num)
    print(has_doubts_text)
