import cv2
import pickle

import communication_pb2
import communication_pb2_grpc

import grpc


class RemoteRecognition:

    def __init__(self):
        self.channel = grpc.insecure_channel('127.0.0.1:5001')
        self.stub = communication_pb2_grpc.dataStub(self.channel)

        self.online = False
        self.tem_classes = []
        self.tem_number = 10
        self.max_len = 0
        self.stable_num = 5
        self.final_classes = ["" for _ in range(10)]

    def run(self, image):
        dumped_image = pickle.dumps(image, 4)
        response = self.stub.recognize(communication_pb2.data_request(b64image=dumped_image))
        boxes = pickle.loads(response.boxes)
        success, classes, boxes = response.success, response.classes, boxes
        success, classes, boxes = self.stable(success, classes, boxes)
        #for _ in range(10 - len(classes)):
        #    classes.append('')
        return success, classes, boxes
    
    def stable(self, success, classes, boxes):

        print(success, len(classes), classes)

        if len(classes) <= 3:
            self.online = False
            self.max_len = 0
            self.tem_classes = []
            self.final_classes = ["" for _ in range(10)]
            return None, ["" for _ in range(10)], []
        else:
            self.online = True

        current_len = len(classes)
        if current_len > self.max_len:
            self.max_len = current_len
        
        classes.extend(["" for _ in range(15 - len(classes))])
        self.tem_classes.insert(0, classes)
        queue_len = len(self.tem_classes)

        if queue_len > self.tem_number:
            self.tem_classes.pop()
        
        if len(self.tem_classes) >= self.stable_num:
            for i in range(self.max_len):
                success = True
                for j in range(1, self.stable_num):
                    if self.tem_classes[0][i] != self.tem_classes[j][i]:
                        success = False
                if success == True:
                    self.final_classes[i] = self.tem_classes[0][i]

        result_class = []
        for i in range(len(self.final_classes)):
            if self.final_classes[i] == "" and i < self.max_len:
                result_class.append("?")
            else:
                result_class.append(self.final_classes[i])

        return True, result_class, boxes


if __name__ == '__main__':
    run()
