from datetime import datetime

import numpy as np
import sklearn
from sklearn.metrics import classification_report
import keras
import matplotlib.pyplot as plt

import os
import seaborn as sns
import tensorboard
import tensorflow as tf
import pathlib

from keras.applications import VGG16, VGG19
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU
from keras.layers import Resizing, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Rescaling
from keras.models import Sequential

from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

import MatrizConfusao
from modelostransferlearning import ModeloTransferLearning

# Dados globais
img_size = (224, 224)
batch_size = 32
epochs = 1


# Carregamento das imagens da amostra
path_treinamento = pathlib.Path(os.path.join("E:/USP_ESALQ", "ProjetoClassificacaoGatos", "Dados", "train"))

path_validacao = pathlib.Path(os.path.join("E:/USP_ESALQ", "ProjetoClassificacaoGatos", "Dados", "valid"))

path_teste = pathlib.Path(os.path.join("E:/USP_ESALQ", "ProjetoClassificacaoGatos", "Dados", "test"))
"""
dados_treinamento = ImageDataGenerator().flow_from_directory(path_treinamento, target_size=img_size)
dados_teste = ImageDataGenerator().flow_from_directory(path_teste, target_size=img_size)
dados_validacao = ImageDataGenerator().flow_from_directory(path_validacao, target_size=img_size)
"""
image_count = len(list(path_treinamento.glob('*/*.jpg')))
print("total de imagens de treino:", image_count)
print("classes:", os.listdir(path_treinamento))

train_dataset = tf.keras.utils.image_dataset_from_directory(
    path_treinamento,
    labels="inferred",
    label_mode="categorical",
    subset="training",
    validation_split=0.2,
    seed=321,
    image_size=(img_size[0], img_size[1]),
    batch_size=batch_size)

test_dataset = tf.keras.utils.image_dataset_from_directory(
        path_teste,
        labels="inferred",
        label_mode="categorical",
        subset=None,
        validation_split=None,
        seed=321,
        image_size=(img_size[0], img_size[1]),
        batch_size=batch_size)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    path_treinamento,
    labels="inferred",
    label_mode="categorical",
    subset="validation",
    validation_split=0.2,
    seed=321,
    image_size=(img_size[0], img_size[1]),
    batch_size=batch_size)

numero_classes = len(train_dataset.class_names)

# Data Augmentation - ImageDataGenerator
gerador_aumento_dados = ImageDataGenerator(
    rotation_range=30,  # rotation
    width_shift_range=0.2,  # horizontal shift
    height_shift_range=0.2,  # vertical shift
    zoom_range=0.2,  # zoom
    horizontal_flip=True  # horizontal flip
)


############################################# MODELO 0 - VGG16 #############################################

vgg16Model = ModeloTransferLearning().novaredevgg16(numero_classes)
#resumo das camadas do modelo
vgg16Model.summary()
# Treinando o modelo
hist_vgg16 = vgg16Model.fit(train_dataset,
                            steps_per_epoch=image_count // batch_size,
                            epochs=epochs,
                            validation_data=validation_dataset,
                            validation_steps=65 // batch_size,
                            )

vgg16Model.save_weights("pesosvgg16.h5")

##matriz de confusao
predicao_vgg16 = vgg16Model.predict(test_dataset)

test_pred = np.argmax(predicao_vgg16, axis=1)

test_labels = np.concatenate([y for x, y in test_dataset], axis=0)

test_labels = np.argmax(test_labels.__array__(), axis=1)

# Relatorio metrico
print(classification_report(test_labels, test_pred, labels=test_dataset.class_names, zero_division=0))

# Calcular the confusion matrix using sklearn.metrics
cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
figure, plt = MatrizConfusao.plot_confusion_matrix(cm, class_names=test_dataset.class_names)
figure.savefig('logs/matrizConfusaoVGG16.png', format='png')
plt.close(figure)

# Precision and Validation Precision
figure = plt.figure(figsize=(8, 8))
plt.title('Precisao VGG-16')
plt.plot(hist_vgg16.history["precision"], label="Dados de Treino")
plt.plot(hist_vgg16.history["val_precision"], label="Dados de Validação")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Precisão')
figure.savefig('logs/precisaoVGG16', format='png')
plt.close(figure)

# Accuracy and Validation Accuracy
figure = plt.figure(figsize=(8, 8))
plt.title('Acurácia VGG-16')
plt.plot(hist_vgg16.history["accuracy"], label="Dados de Treino")
plt.plot(hist_vgg16.history["val_accuracy"], label="Dados de Validação")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Acurácia')
figure.savefig('logs/acuraciaVGG16', format='png')
plt.close(figure)
plt.show()

############################################# MODELO 0 - VGG16 #############################################

############################################# MODELO 1 - VGG19 #############################################

vgg19Model = ModeloTransferLearning().novaredevgg19(numero_classes)
#resumo das camadas do modelo
vgg19Model.summary()
# Treinando o modelo
hist_vgg19 = vgg19Model.fit(train_dataset,
                            steps_per_epoch=image_count // batch_size,
                            epochs=epochs,
                            validation_data=validation_dataset,
                            validation_steps=65 // batch_size)

vgg19Model.save_weights("pesosvgg19.h5")

##matriz de confusao
predicao_vgg19 = vgg19Model.predict(test_dataset, batch_size=batch_size)
test_pred = np.argmax(predicao_vgg19, axis=1)
test_labels = np.concatenate([y for x, y in test_dataset], axis=0)
test_labels = np.argmax(test_labels, axis=1)
#Relatorio
print(classification_report(test_labels, test_pred, labels=test_dataset.class_names, zero_division=0))
# Calcular the confusion matrix using sklearn.metrics
cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
figure, plt = MatrizConfusao.plot_confusion_matrix(cm, class_names=test_dataset.class_names)
figure.savefig('logs/matrizConfusaoVGG19.png', format='png')
plt.close(figure)

# Precision and Validation Precision
plt.plot(hist_vgg19.history["precision"], label="Dados de Treino")
plt.plot(hist_vgg19.history["val_precision"], label="Dados de Validação")
plt.title('Precisão VGG-19')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Precisão')
figure.savefig('logs/precisaoVGG19.png', format='png')
plt.close(figure)

# Accuracy and Validation Accuracy
plt.plot(hist_vgg19.history["accuracy"], label="Dados de Treino")
plt.plot(hist_vgg19.history["val_accuracy"], label="Dados de Validação")
plt.title('Acurácia VGG-16')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Acurácia')
figure.savefig('logs/acuraciaVGG19', format='png')
plt.close(figure)
plt.show()
############################################# MODELO 1 - VGG19 #############################################


############################################# MODELO 2 #############################################

modelo2 = Sequential()
modelo2.add(Resizing(img_size[0], img_size[1], interpolation="bilinear", crop_to_aspect_ratio=True,
                     name='camada_redimensionamento'))

modelo2.add(Rescaling(1. / 255, input_shape=(img_size[0], img_size[1], 3), name='camada de escalonamento'))

modelo2.add(Conv2D(16, 3, padding='same', activation=LeakyReLU(alpha=0.05), name='camada intermediaria 1'))
modelo2.add(MaxPooling2D(pool_size=(2, 2)))

modelo2.add(Conv2D(32, 3, padding='same', activation=LeakyReLU(alpha=0.04), name='camada intermediaria 2'))
modelo2.add(MaxPooling2D(pool_size=(2, 2)))

modelo2.add(Conv2D(64, 3, padding='same', activation=LeakyReLU(alpha=0.01), name='camada intermediaria 3'))

modelo2.add(Conv2D(64, 3, padding='same', activation=LeakyReLU(alpha=0.01), name='camada intermediaria 4'))
modelo2.add(MaxPooling2D(pool_size=(2, 2)))

modelo2.add(Conv2D(128, 3, padding='same', activation=LeakyReLU(alpha=0.01), name='camada intermediaria 5'))

modelo2.add(Conv2D(128, 3, padding='same', activation=LeakyReLU(alpha=0.01), name='camada intermediaria 6'))
modelo2.add(MaxPooling2D(pool_size=(2, 2)))

modelo2.add(Dropout(0.3,name='camada de dropout 30%'),)

modelo2.add(Flatten())

modelo2.add(Dense(128, activation=LeakyReLU(alpha=0.01)))

modelo2.add(Dense(numero_classes, activation='softmax', name='camada de output - previsões'))

modelo2.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy', tf.keras.metrics.Precision()])

modelo2.build(input_shape=(image_count, img_size[0], img_size[1], 3))

modelo2.summary()

history2 = modelo2.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, )
