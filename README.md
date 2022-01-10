# YOLOv4 with post-processing optimized algorithms

## Abstract

No matter how good the model is, it is inevitable that it still cannot solve some exception situations. For example, duplicate bounding boxes or reversed characters.
Therefore, I aim to write algorithms to optimize the final result.

## Introduction

I have implemented two algorithms to optimize yolov4 in character recognition:

1. Error bounding box detection
2. Upside-down characters detection

## Algorithms flow chart
1. Error bounding box detection

<br>![image](https://user-images.githubusercontent.com/56544982/143669533-6ad3ec75-0dc5-4169-8611-a6282046d658.png)

2. Upside-down characters detection

<br>![image](https://user-images.githubusercontent.com/56544982/143669545-e44f7c3e-2766-425a-ba11-9f8fbafbb44e.png)

## Key feature

Normal algorithms only compare anterior and posterior bounding boxes and cannot compare all the duplicate bounding boxes. 

![image](https://user-images.githubusercontent.com/56544982/143669646-9175078c-9404-49a0-92bb-b7ff4fe58d0e.png)

My algorithms can compare all the duplicate bbox and choose the right one in the end.

![image](https://user-images.githubusercontent.com/56544982/143669663-60761717-52d0-448e-8ed4-d280bafa1e51.png)

## Usage
1. This code is based on YOLOv4, so please download the code from here first: https://github.com/miemie2013/Pytorch-YOLOv4
2. All algorithm functions are in yolov4_module.py
3. You can refer to my main code to know how to use those algorithms.

## Demo

![image](https://user-images.githubusercontent.com/56544982/143669714-b851c601-5408-4881-ae39-68146f9ae6da.png)

![image](https://user-images.githubusercontent.com/56544982/143669733-fd1a7cf0-0cb8-42d4-a5f1-52cd3b31f0ae.png)



