#!/usr/bin/env python
# coding: utf-8

# # Obtención de las mascaras creadas en Labelbox

# ## Importación Librerias
import json
import numpy as np

from PIL import Image
import requests
from io import BytesIO
import os
import sys
from tqdm import tqdm

if os.environ.get('SIMEPU_DATA') is not None:
    SIMEPU_DATA_PATH = os.environ.get('SIMEPU_DATA')

# ## Grietas Longitudinales

download = input("Quieres descargar las  mascaras de las grietas longitudinales? (si/no):")
if download == "si":
    with open("datos/final/longitudinales-2020-09-17-980.json") as json_file:
        datos_longitudinales = json.load(json_file)

    save_path = os.path.join(SIMEPU_DATA_PATH, "Mascaras", "Grietas longitudinales")
    os.makedirs(save_path, exist_ok=True)
    print("Decargando Grietas Longitudinales...")
    for item in tqdm(datos_longitudinales, file=sys.stdout):
        if len(item["Label"]):
            nombre = item["External ID"]
            mascara_url = item["Label"]["objects"][0]["instanceURI"]
            print(mascara_url)
            response = requests.get(mascara_url)
            img = Image.open(BytesIO(response.content))

            img = np.where(np.array(img)[..., 0:3] > 0.5, 255, 0).astype(np.uint8)
            img_file = Image.fromarray(img)
            img_file.save(os.path.join(save_path, nombre))


# ## Grietas Transversales

download = input("Quieres descargar las  mascaras de las grietas transversales? (si/no):")
if download == "si":
    with open("datos/final/transversales-2020-09-21-584.json") as json_file:
        datos_transversales = json.load(json_file)

    save_path = os.path.join(SIMEPU_DATA_PATH, "Mascaras", "Grietas transversales")
    os.makedirs(save_path, exist_ok=True)
    print("Decargando Grietas Transversales...")
    for item in tqdm(datos_transversales, file=sys.stdout):
        if len(item["Label"]):
            nombre = item["External ID"]
            mascara_url = item["Label"]["objects"][0]["instanceURI"]

            response = requests.get(mascara_url)
            img = Image.open(BytesIO(response.content))

            img = np.where(np.array(img)[..., 0:3] > 0.5, 255, 0).astype(np.uint8)
            img_file = Image.fromarray(img)
            img_file.save(os.path.join(save_path, nombre))


# ## Huecos

download = input("Quieres descargar las mascaras de los huecos? (si/no):")
if download == "si":
    with open("datos/final/huecos-2020-09-21-494.json") as json_file:
        datos_huecos = json.load(json_file)
        
    save_path = os.path.join(SIMEPU_DATA_PATH, "Mascaras", "Huecos")
    os.makedirs(save_path, exist_ok=True)
    print("Decargando Huecos...")
    for item in tqdm(datos_huecos, file=sys.stdout):
        if len(item["Label"]):
            nombre = item["External ID"]
            mascara_url = item["Label"]["objects"][0]["instanceURI"]

            response = requests.get(mascara_url)
            img = Image.open(BytesIO(response.content))

            img = np.where(np.array(img)[..., 0:3] > 0.5, 255, 0).astype(np.uint8)
            img_file = Image.fromarray(img)
            img_file.save(os.path.join(save_path, nombre))

# ## Parcheos

download = input("Quieres descargar las mascaras de los parcheos? (si/no):")
if download == "si":
    with open("datos/final/parcheo-2020-09-21-386.json") as json_file:
        datos_parcheos = json.load(json_file)
        
    save_path = os.path.join(SIMEPU_DATA_PATH, "Mascaras", "Parcheo")
    os.makedirs(save_path, exist_ok=True)
    print("Decargando Parcheos...")
    for item in tqdm(datos_parcheos, file=sys.stdout):
        if len(item["Label"]):
            nombre = item["External ID"]
            mascara_url = item["Label"]["objects"][0]["instanceURI"]

            response = requests.get(mascara_url)
            img = Image.open(BytesIO(response.content))

            img = np.where(np.array(img)[..., 0:3] > 0.5, 255, 0).astype(np.uint8)
            img_file = Image.fromarray(img)
            img_file.save(os.path.join(save_path, nombre))