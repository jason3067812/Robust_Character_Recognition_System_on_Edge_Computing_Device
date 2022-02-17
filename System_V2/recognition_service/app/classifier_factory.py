from tensorflow.keras import models


def build_efficientnet_classifier(input_size, weights_path):
    custom_objects = {'weighted_binary_crossentropy': None}
    efficientnet = models.load_model(weights_path, custom_objects=custom_objects)
    return efficientnet


def build_reverse_classifier(input_size, weights_path):
    mobilenet = models.load_model(weights_path)
    return mobilenet
