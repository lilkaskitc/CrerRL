import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Activation, Input, Dense, GlobalAveragePooling2D, BatchNormalization, Flatten

from vit_keras import vit

### Design Interval SoftMax Activation Function ###
@keras.saving.register_keras_serializable(package="my_package", name="IntSoftMax")
def IntSoftMax(inputs):
  # Extract number of classes
  Nc = int(inputs.shape[-1]/2)

  # Extract center and the radius
  center = inputs[:, :Nc]
  radius = inputs[:, Nc:]

  # Ensure the nonnegativity of radius
  radius_nonneg = tf.math.softplus(radius)

  # Compute upper and lower probabilities
  lo = K.exp(center-radius_nonneg) / (K.sum(K.exp(center), axis=-1, keepdims=True) - K.exp(center) + K.exp(center-radius_nonneg))
  hi = K.exp(center+radius_nonneg) / (K.sum(K.exp(center), axis=-1, keepdims=True) - K.exp(center) + K.exp(center+radius_nonneg))


  # Generata output
  output = tf.concat([lo, hi], axis=-1)

  return output


def CreNetVGG16(input_shape, classes, weights):
    base = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    inputs = Input(input_shape)
    x = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
    x = base(x)
    # x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(units=1024, activation='relu')(x)
    x = Dense(units=512, activation='relu')(x) 
    x = Dense(units=2*classes, activation=None)(x)
    x = BatchNormalization()(x)
    outputs = Activation(IntSoftMax)(x)

    model = keras.Model(inputs, outputs, name='CreNet_VGG16')

    return model

def CreNetRES50(input_shape, classes, weights):
    base = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=weights, input_shape=(224, 224, 3), classes=classes)
    inputs = Input(input_shape)
    x = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
    x = base(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(units=1024, activation='relu')(x)
    x = Dense(units=512, activation='relu')(x) 
    x = Dense(units=2*classes, activation=None)(x)
    x = BatchNormalization()(x)
    outputs = Activation(IntSoftMax)(x)

    model = keras.Model(inputs, outputs, name='CreNet_RES50')

    return model




def CreNetVITBase(input_shape, classes, weights):
    base = vit.vit_b16(image_size = 224, pretrained = True, include_top = False, pretrained_top = False)
    inputs = Input(input_shape)
    x = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
    x = base(x)
    # x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(units=1024, activation='relu')(x)
    x = Dense(units=512, activation='relu')(x) 
    x = Dense(units=2*classes, activation=None)(x)
    x = BatchNormalization()(x)
    outputs = Activation(IntSoftMax)(x)

    model = keras.Model(inputs, outputs, name='CreNet_VIT')

    return model