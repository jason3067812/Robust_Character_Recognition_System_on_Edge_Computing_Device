# Robust Character Recognition System on Nvidia AGX

## Abstract
This project is a Industry and Academia Cooperation from ChinaSteel Inc. and Yaun Ze University that aims to help identify characters on steel plates.

## Introduction

This project includes two different methods (System_V1 and System_V2). For details, please click on the above two folders and read the README inside them.

## Dataset Exhibition

1. Steel Plates:

<img src="https://user-images.githubusercontent.com/56544982/154431220-e3a90c64-1962-44c8-b298-56d27993de98.png" alt="Cover" width="60%"/>

2. Steel Coils:

<img src="https://user-images.githubusercontent.com/56544982/154430938-994a7c87-5ea6-4c39-80fd-401f922628a5.png" alt="Cover" width="60%"/>

## System Hardware Setting Exhibition

<img src="https://user-images.githubusercontent.com/56544982/154511296-12788e64-a9de-4bfb-9992-205174f549e3.png" alt="Cover" width="60%"/>

## System Interface Exhibition

<img src="https://user-images.githubusercontent.com/56544982/154509701-c7a5194c-625f-4dd8-ad45-58f7447a11f6.png" alt="Cover" width="60%"/>

<img src="https://user-images.githubusercontent.com/56544982/154509721-c07edd75-5076-46ea-9ed6-e3a3eb2f0670.png" alt="Cover" width="60%"/>

## Testing Results
By comparing system version 1 and version 2:

1. Testing steel plates:

<img src="https://user-images.githubusercontent.com/56544982/154624533-a2ccbf4d-3d08-4c68-9adf-f40111f96923.png" alt="Cover" width="60%"/>

2. Testing steel coils by transfer learning (measure system's universality):

<img src="https://user-images.githubusercontent.com/56544982/154624603-5ba20a89-40a5-4709-86ff-d28b9b6f5389.png" alt="Cover" width="60%"/>

## Demo Video

https://www.youtube.com/watch?v=7dJ7rLHM37Y

## Platform/Requirement
1. Edge Computing Device: Nvidia AGX
2. Camera: Basler Industrial Camera (acA2440-20gm)
3. Operating System: Linux
4. Programming Language: Python
5. Environment: Pytorch for System_V1 and Tensorflow for System_V2

## Contributor
Really thanks to all of them, I learned a lot from them!
- Supervisor: Professor Andrew Lin (andrewlin@g.yzu.edu.tw)
- Teammates: Stephen Li (cyli09701225@gmail.com), Jeffery Chen (s1063715@mail.yzu.edu.tw), Jenna Weng (JennaWeng0621@gmail.com)

## Reference
1. https://github.com/miemie2013/Pytorch-YOLOv4
2. https://github.com/clovaai/CRAFT-pytorch
3. https://grpc.io/docs/languages/python/basics/
4. https://shengyu7697.github.io/python-tcp-socket/
5. https://www.baslerweb.com/cn/sales-support/downloads/software-downloads/#type=pylonsoftware;language=all;version=all
6. https://github.com/basler/pypylon/releases/tag/1.6.0
