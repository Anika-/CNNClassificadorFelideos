import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.applications import VGG16, VGG19
from keras.layers import Dense


class ModeloTransferLearning(object):

    def novaredevgg16(self, numero_classes):

        vgg16 = VGG16()
        vgg16model = Sequential()

        for i in range(len(vgg16.layers) - 1):
            vgg16model.add(vgg16.layers[i])

        # Fecha as camadas do vgg16 para impedir alterações nos pesos intermediários
        for layers in vgg16model.layers:
            layers.trainable = False

        # Classes customizadas na ultima camada
        vgg16model.add(Dense(numero_classes, activation="softmax", name="predicao_vgg16"))

        vgg16model.compile(loss="categorical_crossentropy",
                           optimizer="rmsprop",
                           metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return vgg16model

    def novaredevgg19(self, numero_classes):

        vgg19 = VGG19()
        vgg19model = Sequential()

        for i in range(len(vgg19.layers) - 1):
            vgg19model.add(vgg19.layers[i])

        # Fecha as camadas do vgg16 para impedir alterações nos pesos intermediários
        for layers in vgg19model.layers:
            layers.trainable = False

        # Classes customizadas na ultima camada
        vgg19model.add(Dense(numero_classes, activation="softmax", name="predicao_vgg19"))

        vgg19model.compile(loss="categorical_crossentropy",
                           optimizer="rmsprop",
                           metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return vgg19model
