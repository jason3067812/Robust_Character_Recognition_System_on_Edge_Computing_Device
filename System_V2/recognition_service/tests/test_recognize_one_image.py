import logging

from app.recognition import Recognition

import cv2

from matplotlib import pyplot as plt

FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.INFO)


def __plot_result_image(image, boxes, classes):
    for index in range(len(boxes)):
        top_left = (boxes[index][0][0], boxes[index][0][1])
        down_left = (boxes[index][3][0], boxes[index][3][1])
        down_right = (boxes[index][2][0], boxes[index][2][1])
        cv2.rectangle(image, top_left, down_right, (255, 0, 0), 2)
        cv2.putText(
            image,
            classes[index],
            down_left,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    plt.savefig("result.jpg")
    logging.info("recognized result is saved to result.jpg")


if __name__ == "__main__":
    recognition_service = Recognition()

    image_path = "../datasets/end_to_end/testing_data/images/2020_08_13_02_20_30.bmp"
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)

    success, classes, boxes = recognition_service.recognize(image)

    logging.info("recognized class: {}".format(classes))
    __plot_result_image(image, boxes, classes)
