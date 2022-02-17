# System Version 2: Craft + MobileNet + EfficientNet

## Usage
This system has two modules (two folders):

- recognition_service: For the recognition service based on Tensorflow, please set it up first and run it.
- cam_ui: A UI interface for reading and displaying results for the camera.

## Introduction
- Backend models:
  - a. CRAFT: detecting the area of characters
  - b. MobileNet: recognizing if characters are reversed or not
  - c. EfficientNet: identifying the character
- Frontend: User Interface
- Transmission protocal: gRPC

