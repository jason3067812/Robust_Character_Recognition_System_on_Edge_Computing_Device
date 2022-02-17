import logging

from app.classifier_factory import build_efficientnet_classifier, build_reverse_classifier
from app.craft_network import build_keras_model, normalize_image
from app.utils import get_det_boxes

import cv2

import numpy as np

import tensorflow as tf


chars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G",
         "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


class Recognition:

    def __init__(self):
        self.text_threshold = 0.4
        self.link_threshold = 0.4
        self.low_text = 0.4
        self.chars_map = {index: value for index, value in enumerate(chars)}
        weights_path = "models/csc_sheet_fine_tune_20-0.0366277471.hdf5"
        self.__craft_model = build_keras_model("vgg", (512, 512, 3), weights_path)
        weights_path = "models/mobilenet_difficult_and_reverse_48-0.2623.hdf5"
        self.__reverse_classifier = build_reverse_classifier((64, 64, 1), weights_path)
        weights_path = "models/efficientNetb3_weighted_18-0.9985.hdf5"
        self.__classifier = build_efficientnet_classifier((64, 64, 1), weights_path)

    def recognize(self, image):
        image = self.__preprocess_image(image)
        bounding_boxes = self.__locate(image.copy())
        chars = self.__crop_region(image, bounding_boxes)

        reverse = self.__classify_reverse(chars.copy())
        if reverse is True:
            chars = self.__reverse_char_image(chars)

        success, classes = self.__classify_char(chars.copy())
        return success, classes, bounding_boxes

    @staticmethod
    def __preprocess_image(image):
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
        return image

    def __locate(self, image):
        image = normalize_image(image)
        image = image[np.newaxis]
        result = self.__craft_model.predict(image)
        bounding_boxes = self.__get_bounding_box(result)
        return bounding_boxes

    def __get_bounding_box(self, heatmap_result):
        score_text = heatmap_result[0, :, :, 0]
        score_link = heatmap_result[0, :, :, 1]
        boxes = get_det_boxes(
            score_text,
            score_link,
            self.text_threshold,
            self.link_threshold,
            self.low_text,
        )
        boxes = np.array(boxes, dtype=np.int32) * 2
        boxes = np.sort(boxes, axis=0)
        boxes[boxes > 512] = 512
        boxes[boxes < 0] = 0
        return boxes

    @staticmethod
    def __crop_region(image, boxes):
        char_num = len(boxes)
        char_images = np.zeros((char_num, 64, 64, 1))
        for i in range(char_num):
            box = boxes[i]
            char_image = image[box[0][1]:box[2][1], box[0][0]:box[2][0]]
            char_image = tf.image.resize(char_image, (64, 64))
            char_image = tf.image.rgb_to_grayscale(char_image)
            char_images[i] = char_image
        return char_images

    def __classify_reverse(self, chars):
        results = self.__get_char_reverse_result(chars)
        reserve = self.__judge_reserve(results)
        logging.debug("reverse result: {}".format(reserve))
        return reserve

    def __get_char_reverse_result(self, tem_image):
        tem_image = tem_image / 255
        result = self.__reverse_classifier.predict(tem_image)
        logging.debug("reverse confidence: {}".format(result))
        return result

    def __judge_reserve(self, result):
        reserve = []
        for i in range(len(result[0])):
            if result[1][i][0] < 0.5:
                reserve.append(result[0][i][0])
        mean = np.sum(reserve) / len(reserve)
        if mean > 0.5:
            return True
        else:
            return False

    def __reverse_char_image(self, chars):
        for i in range(len(chars)):
            chars[i] = cv2.flip(chars[i], -1)[:, :, np.newaxis]
        return chars

    def __classify_char(self, chars):
        result = self.__get_char_confidences(chars)

        successes = []
        classes = []
        for index in range(len(chars)):
            success, cla = self.__judge_char_class(result[index])
            successes.append(success)
            classes.append(cla)
        success = all(successes)
        logging.debug("char recognition is success: {}".format(success))
        logging.debug("char recognized classes: {}".format(classes))
        return success, classes

    def __get_char_confidences(self, sub_item):
        result = self.__classifier.predict(sub_item)
        logging.debug("char class confidence: {}".format(result))
        return result

    def __judge_char_class(self, result):
        top_2 = result[(-result).argsort()[:2]]
        index = np.argmax(result)
        if np.abs(top_2[0] - top_2[1]) < 0.3:
            success = False
        elif result[index] < 0.8:
            success = False
        else:
            success = True
        cla = self.chars_map[index]
        return success, cla
