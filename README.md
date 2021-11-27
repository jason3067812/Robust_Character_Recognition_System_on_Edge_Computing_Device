# Yolov4_with_post-processing_optimized_ralgorithm

## Abstract
No matter how good the model is, it is inevitable that it still cannot solve some exception situations. For example, duplicate bounding boxes or reversed characters.
Therefore, I aim to write algorithms to optimize the final result.

## Introduction
I have implemented two algorithms to optimize yolov4 in character recognition:
1. Error bounding box detection
2. Upside-down characters detection

## Usage
1. This code is based on yolov4, so please download the code from here first: https://github.com/miemie2013/Pytorch-YOLOv4
2. All algorithm functions are in yolov4_module.py
3. You can refer to my main code to know how to use those algorithms.

