
# # Projeto Cats and Dogs.ipynb 
# # Que estava no google Colab 
# # talvez não esteja mais lá https://colab.research.google.com/drive/10Tl1sTwWFVBpmFiEzOlQuwMa-v9hnDf4




# import os
# !wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip -O dataset.zip

    
# # -----------------------------------------------------------------------------
# !unzip -q dataset.zip
# !ls

# # -----------------------------------------------------------------------------

# # # dezipa em data
# # !unzip archive.zip -d /content/data

# num_skipped = 0
# for folder_name in ("Cat", "Dog"):
#   folder_path = os.path.join("PetImages", folder_name)
#   for fname in os.listdir(folder_path):
#     fpath = os.path.join(folder_path, fname)
#     try:
#       fobj = open(fpath, "rb")
#       is_jfif = b"JFIF" in fobj.peek(10)
#     finally:
#       fobj.close()

#     if not is_jfif:
#       num_skipped += 1
#       os.remove(fpath)
# print("Deleted images: {}".format(str(num_skipped)))

# # -----------------------------------------------------------------------------

# import tensorflow as tf
# from tensorflow.keras.preprocessing import image_dataset_from_directory

# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras import layers, models


# import numpy as np
# from PIL import Image

# from sklearn.model_selection import train_test_split
# import shutil
# import random

# # -----------------------------------------------------------------------------

# caminho_processadas_cat = "/content/PetImages/Cat"
# caminho_processadas_dog = "/content/PetImages/Dog"
# caminho_train = "/content/data/train"
# caminho_test = "/content/data/test"

# # Cria as pastas de trino e de teste dos cats e dos Dogs
# os.makedirs(os.path.join(caminho_train, 'Cat'), exist_ok= True)
# os.makedirs(os.path.join(caminho_train, 'Dog'), exist_ok= True)
# os.makedirs(os.path.join(caminho_test, 'Cat'), exist_ok= True)
# os.makedirs(os.path.join(caminho_test, 'Dog'), exist_ok= True)

# # -----------------------------------------------------------------------------
# def is_image(filename):
#     try:
#         Image.open(filename).verify()
#         return True
#     except:
#         return False

# # Verifica se a imagem é uma imagem, retorna o tipo da imagem  cat ou dog
# def listar_imagens_tipo():
#     imagens_tipo = []

#     imagens_cat = os.listdir(caminho_processadas_cat)
#     for imagem in imagens_cat:
#         imagens_path = os.path.join(caminho_processadas_cat, imagem)
#         if os.path.isfile(imagens_path) and is_image(imagens_path):
#             imagens_tipo.append((imagens_path, 'Cat'))

#     imagens_dog = os.listdir(caminho_processadas_dog)
#     for imagem in imagens_dog:
#         imagens_path = os.path.join(caminho_processadas_dog, imagem)
#         if os.path.isfile(imagens_path) and is_image(imagens_path):
#             imagens_tipo.append((imagens_path, 'Dog'))

#     return imagens_tipo
#   # divide as imagens em treino e teste
# def dividir_imagens(imagens_tipo,slit_radio=0.8):
#     imagens_path, labels = zip(*imagens_tipo)

#     X_train, X_test, y_train, y_test = train_test_split(imagens_path, labels, test_size=1-slit_radio, random_state=42,shuffle=True)

#     for imagens_path, tipo in zip(X_train, y_train):
#       destino = os.path.join(caminho_train, tipo, os.path.basename(imagens_path))
#       if os.path.exists(imagens_path):
#         shutil.move(imagens_path, destino)
#       else:
#         print(f"A imagem {imagens_path} não existe.")

#     for imagens_path, tipo in zip(X_test, y_test):
#       destino = os.path.join(caminho_test, tipo, os.path.basename(imagens_path))
#       if os.path.exists(imagens_path):
#         shutil.move(imagens_path, destino)
#       else:
#         print(f"A imagem {imagens_path} não existe.")
#     print("Imagens dividas as imagens em treino e teste")
# # -----------------------------------------------------------------------------

# imagens_tipo = listar_imagens_tipo()
# # -----------------------------------------------------------------------------

# dividir_imagens(imagens_tipo)
# # -----------------------------------------------------------------------------
# train_dir = '/content/data/train'
# test_dir = '/content/data/test'

# # -----------------------------------------------------------------------------
# # Configuração de parâmetros
# BATCH_SIZE = 32
# IMG_SIZE = (224, 224)


# train_dataset = image_dataset_from_directory(
#     train_dir,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE
# )

# val_dataset = image_dataset_from_directory(
#     test_dir,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE
# )
# # -----------------------------------------------------------------------------

# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras import layers, models

# # Construção do modelo com Transfer Learning
# base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
# base_model.trainable = False # Congela os pesos do modelo pré-treinado

# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(128, activation="relu"),
#     layers.Dropout(0.5),
#     layers.Dense(1, activation="sigmoid") # Saída binária (0 = gato, 1 = cachorro)
# ])

# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# # Treinamento do modelo
# model.fit(train_dataset, validation_data=val_dataset, epochs=5)
# # -----------------------------------------------------------------------------
# # Salvar modelo treinado
# model.save("modelo_CatsAndDogs_mobilenetv2.h5")
# # -----------------------------------------------------------------------------