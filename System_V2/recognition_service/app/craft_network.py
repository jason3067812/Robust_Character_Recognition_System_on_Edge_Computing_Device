import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def normalize_image(image):
    image = image.astype('float32')
    mean = np.array([0.485, 0.456, 0.406])
    variance = np.array([0.229, 0.224, 0.225])

    image -= mean * 255
    image /= variance * 255
    return image


def upconv(x, n, filters):
    x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, name=f'upconv{n}.conv.0')(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f'upconv{n}.conv.1')(x)
    x = layers.Activation('relu', name=f'upconv{n}.conv.2')(x)
    x = layers.Conv2D(
        filters=filters // 2,
        kernel_size=3,
        strides=1,
        padding='same',
        name=f'upconv{n}.conv.3',
    )(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f'upconv{n}.conv.4')(x)
    x = layers.Activation('relu', name=f'upconv{n}.conv.5')(x)
    return x


def make_vgg_block(x, filters, n, prefix, pooling=True):
    x = layers.Conv2D(
        filters=filters,
        strides=(1, 1),
        kernel_size=(3, 3),
        padding='same',
        name=f'{prefix}.{n}',
    )(x)
    x = layers.BatchNormalization(
        momentum=0.1,
        epsilon=1e-5,
        axis=-1,
        name=f'{prefix}.{n+1}',
    )(x)
    x = layers.Activation('relu', name=f'{prefix}.{n+2}')(x)
    if pooling:
        x = layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='valid',
            strides=(2, 2),
            name=f'{prefix}.{n+3}',
        )(x)
    return x


class UpsampleLike(layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor."""

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == 'channels_first':
            raise NotImplementedError
        else:
            return tf.compat.v1.image.resize_bilinear(
                source,
                size=(target_shape[1], target_shape[2]),
                half_pixel_centers=True,
            )

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_first':
            raise NotImplementedError
        else:
            return (input_shape[0][0], ) + input_shape[1][1:3] + (input_shape[0][-1], )


def build_vgg_backbone(inputs):
    x = make_vgg_block(inputs, filters=64, n=0, pooling=False, prefix='basenet.slice1')
    x = make_vgg_block(x, filters=64, n=3, pooling=True, prefix='basenet.slice1')
    x = make_vgg_block(x, filters=128, n=7, pooling=False, prefix='basenet.slice1')
    x = make_vgg_block(x, filters=128, n=10, pooling=True, prefix='basenet.slice1')
    x = make_vgg_block(x, filters=256, n=14, pooling=False, prefix='basenet.slice2')
    x = make_vgg_block(x, filters=256, n=17, pooling=False, prefix='basenet.slice2')
    x = make_vgg_block(x, filters=256, n=20, pooling=True, prefix='basenet.slice3')
    x = make_vgg_block(x, filters=512, n=24, pooling=False, prefix='basenet.slice3')
    x = make_vgg_block(x, filters=512, n=27, pooling=False, prefix='basenet.slice3')
    x = make_vgg_block(x, filters=512, n=30, pooling=True, prefix='basenet.slice4')
    x = make_vgg_block(x, filters=512, n=34, pooling=False, prefix='basenet.slice4')
    x = make_vgg_block(x, filters=512, n=37, pooling=False, prefix='basenet.slice4')
    x = make_vgg_block(x, filters=512, n=40, pooling=True, prefix='basenet.slice4')
    vgg = models.Model(inputs=inputs, outputs=x)
    return [
        vgg.get_layer(slice_name).output for slice_name in [
            'basenet.slice1.12',
            'basenet.slice2.19',
            'basenet.slice3.29',
            'basenet.slice4.38',
        ]
    ]


def build_keras_model(backbone_name, input_size, weights_path=None):
    inputs = layers.Input(input_size)

    if backbone_name == 'vgg':
        s1, s2, s3, s4 = build_vgg_backbone(inputs)
    else:
        raise NotImplementedError

    s5 = layers.MaxPooling2D(pool_size=3, strides=1, padding='same', name='basenet.slice5.0')(s4)
    s5 = layers.Conv2D(
        1024,
        kernel_size=(3, 3),
        padding='same',
        strides=1,
        dilation_rate=6,
        name='basenet.slice5.1',
    )(s5)
    s5 = layers.Conv2D(1024, kernel_size=1, strides=1, padding='same', name='basenet.slice5.2')(s5)

    y = layers.Concatenate()([s5, s4])
    y = upconv(y, n=1, filters=512)
    y = UpsampleLike()([y, s3])
    y = layers.Concatenate()([y, s3])
    y = upconv(y, n=2, filters=256)
    y = UpsampleLike()([y, s2])
    y = layers.Concatenate()([y, s2])
    y = upconv(y, n=3, filters=128)
    y = UpsampleLike()([y, s1])
    y = layers.Concatenate()([y, s1])
    ft = upconv(y, n=4, filters=64)

    y = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', name='conv_cls.0')(ft)
    y = layers.Activation('relu', name='conv_cls.1')(y)
    y = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', name='conv_cls.2')(y)
    y = layers.Activation('relu', name='conv_cls.3')(y)
    y = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', name='conv_cls.4')(y)
    y = layers.Activation('relu', name='conv_cls.5')(y)
    y = layers.Conv2D(filters=16, kernel_size=1, strides=1, padding='same', name='conv_cls.6')(y)
    y = layers.Activation('relu', name='conv_cls.7')(y)
    y = layers.Conv2D(filters=2, kernel_size=1, strides=1, padding='same', name='conv_cls.8')(y)

    # load pretrained weight
    model = models.Model(inputs=inputs, outputs=y)

    if weights_path is not None:
        model.load_weights(weights_path)

    return model
