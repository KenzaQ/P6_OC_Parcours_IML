# Importation des librairies

# coding: utf-8

from numpy import load
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps 
from PIL import ImageFilter
from keras.applications.vgg16 import preprocess_input
from tensorflow import keras

# Récupération des données

## Chemin de l'image
chemin_image=input("Quel est le chemin de la photo : ")

## Chargement du modèle
modele = keras.models.load_model('best_model_images.h5')

## Chargement de la liste des races de chien
liste_races_chien = np.load('liste_races_chien.npy')

# Prédiction de la race du chien à partir d'une image

def prediction_image(chemin_image, modele, liste_races_chien):
  ## Traitements de l'image
  img = Image.open(chemin_image).convert('RGB')
  img = img.resize((224, 224))
  img = ImageOps.equalize(img, mask = None)
  img = img.rotate(0, resample=Image.BILINEAR)
  img_traitee = img.filter(ImageFilter.BoxBlur(0))
  img_traitee.show()

  ## Preprocessing de l'image
  img_process = np.array(img_traitee).astype(np.float32)
  img_process = img_process.reshape((1, 224, 224, 3))
  img_process = preprocess_input(img_process)

  ## Prédiction de l'image
  predictions = modele.predict(img_process)
  liste_races_chien = liste_races_chien
  return 'La race du chien sur l\'image la plus probable est : {} avec une probabilité de {}'.format(liste_races_chien[np.argmax(predictions[0])], np.max(predictions[0]))

print(prediction_image(chemin_image, modele, liste_races_chien))
