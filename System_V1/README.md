# System 1 : YOLOv4 with post-processing algorithms

## Abstract

No matter how good the model is, it is inevitable that it still cannot solve some exception situations. For example, the appearance of duplicate bounding boxes when the training dataset is not enough in the initial. Therefore, we aim to write robust algorithms to optimize the final result and help increase the recognition accuracy in the initial.

## Introduction

a. Backend model (YOLOv4):
  - Function:
    1: First detecting the area of characters then 
    2: Then identifying the character in the target area
    
b. Backend post-processing algorithms:
  - We have implemented two algorithms to optimize YOLOv4 in character recognition:
    1. Error bounding box detection
    2. Upside-down characters detection

c. Frontend: Implementing a user interface via tkinter

d. Transmission protocal: Sending images and recognition results via Socket

## Post-processing algorithms flow chart
1. Error bounding boxes detection

<img src="https://user-images.githubusercontent.com/56544982/143669533-6ad3ec75-0dc5-4169-8611-a6282046d658.png" alt="Cover" width="50%"/>

2. Upside-down characters detection

<img src="https://user-images.githubusercontent.com/56544982/143669545-e44f7c3e-2766-425a-ba11-9f8fbafbb44e.png" alt="Cover" width="50%"/>

## Error bounding box detection detail

Normal algorithm only compare anterior and posterior bounding boxes and cannot compare all the duplicate bounding boxes (when there are more than 2 bounding boxes on the same object). 

<img src="https://user-images.githubusercontent.com/56544982/143669646-9175078c-9404-49a0-92bb-b7ff4fe58d0e.png" alt="Cover" width="50%"/>

Therefore I developed an algorithm which can compare all duplicate bboxes and select the only right one in the end (no matter how many boxes appear).

<img src="https://user-images.githubusercontent.com/56544982/143669663-60761717-52d0-448e-8ed4-d280bafa1e51.png" alt="Cover" width="50%"/>

## Training tutorial
1. This code is based on YOLOv4, so please download the code from here first: https://github.com/miemie2013/Pytorch-YOLOv4
2. Download and follow the content in my YOLOv4 environment setup tutorial.pptx step-by-step to finish all pre-processing works
3. Then start training your own dataset

## Predicting tutorial
1. Firstly, setup your camera and connect the cable
2. Secoondly, run final_system.py
3. Thirdly, run cam_UI_v2.py

## Other file function introduction
1. final_module.py: all of algorithms are in this file
2. final_path.py: when you only want to run a image file for testing, you can use this code

## Demo

<img src="https://user-images.githubusercontent.com/56544982/143669714-b851c601-5408-4881-ae39-68146f9ae6da.png" alt="Cover" width="50%"/>

<img src="https://user-images.githubusercontent.com/56544982/143669733-fd1a7cf0-0cb8-42d4-a5f1-52cd3b31f0ae.png" alt="Cover" width="50%"/>




