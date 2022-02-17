import logging
import pickle
from concurrent import futures

from app import communication_pb2
from app import communication_pb2_grpc
from app.recognition import Recognition

import grpc

import numpy as np


FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


class ServerGreeter(communication_pb2_grpc.dataServicer):

    def __init__(self):
        self.recognition_service = Recognition()

    def recognize(self, request, context):
        logging.info("A recognition request was received.")

        image = pickle.loads(request.b64image)

        try:
            success, classes, boxes = self.recognition_service.recognize(image)
        except Exception as err:
            logging.exception(err)
            success, classes, boxes = False, np.array([]), np.array([])

        dumped_boxes = pickle.dumps(boxes, 4)

        return communication_pb2.data_reply(success=success, classes=classes, boxes=dumped_boxes)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    communication_pb2_grpc.add_dataServicer_to_server(ServerGreeter(), server)
    server.add_insecure_port('[::]:5001')
    server.start()
    logging.info("recognition service started.")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
