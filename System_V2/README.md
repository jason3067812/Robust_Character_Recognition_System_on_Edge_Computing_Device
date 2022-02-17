# System Version 2: Craft + MobileNet + EfficientNet

## Usage
This system has two modules (two folders):

- recognition_service: For the recognition service based on Tensorflow, please set it up first and run it.
- cam_ui: A UI interface for reading and displaying results for the camera.

## Introduction
- Backend models:
  1. CRAFT: detecting the area of characters
  2. MobileNet: recognizing if characters are reversed or not
  3. EfficientNet: identifying the character
- Frontend: User Interface
- Transmission protocal: gRPC

## System architecture

![image](https://user-images.githubusercontent.com/56544982/154428730-f4c2a57b-555a-49ba-a6fa-e8869a1408b6.png)




