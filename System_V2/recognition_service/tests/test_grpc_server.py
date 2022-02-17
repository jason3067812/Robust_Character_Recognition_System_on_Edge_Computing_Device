import cv2
import pickle

from app import communication_pb2
from app import communication_pb2_grpc

import grpc


def run():
    channel = grpc.insecure_channel('127.0.0.1:5001')
    stub = communication_pb2_grpc.dataStub(channel)

    # 這裡換成照片機讀到的檔案
    image = cv2.imread("../datasets/end_to_end/testing_data/images/2020_08_13_02_20_30.bmp")
    # 這裡換成照片機讀到的檔案

    dumped_image = pickle.dumps(image, 4)
    response = stub.recognize(communication_pb2.data_request(b64image=dumped_image))
    boxes = pickle.loads(response.boxes)

    # 答案匯出現在這邊
    print(response.success)  # 如果照片有字元信心度不足，或兩種類別信心度太接近。這裡會是 False
    print(response.classes)  # [label * n char]
    print(boxes)             # [(4, 2) * n char]


if __name__ == '__main__':
    run()
