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

# ## Grietas Longitudinales

with open("datos/longitudinales-2020-09-08-451.json") as json_file:
    datos_longitudinales = json.load(json_file)

if os.environ.get('SIMEPU_DATA') is not None:
    SIMEPU_DATA_PATH = os.environ.get('SIMEPU_DATA')

save_path = os.path.join(SIMEPU_DATA_PATH, "Mascaras", "Grietas longitudinales")
os.makedirs(save_path, exist_ok=True)
print("Decargando Grietas Longitudinales...")
for item in tqdm(datos_longitudinales, file=sys.stdout):
    if len(item["Label"]):
        nombre = item["External ID"]
        mascara_url = item["Label"]["objects"][0]["instanceURI"]

        response = requests.get(mascara_url)
        img = Image.open(BytesIO(response.content))

        img = np.where(np.array(img)[..., 0:3] > 0.5, 255, 0).astype(np.uint8)
        img_file = Image.fromarray(img)
        img_file.save(os.path.join(save_path, nombre))
