import urllib.request
import zipfile
import os
url = 'https://syncandshare.lrz.de/dl/fiVRr3EeLiGxcoHXmW5yoWHq/models.zip'

print('Dlownloaing models (This might take time)')
urllib.request.urlretrieve(url, 'models.zip')
with zipfile.ZipFile('models.zip', 'r') as zip_ref:
    print("extracting models")
    zip_ref.extractall('models')
os.remove("models.zip")

with zipfile.ZipFile('models/stargan_models_rel.zip', 'r') as zip_ref:
    print("extracting relative stargan_models")
    zip_ref.extractall('conditional_models/aligned_models/stargan_relative_distance/')
os.remove("stargan_models_rel.zip")

with zipfile.ZipFile('models/stargan_models_abs.zip', 'r') as zip_ref:
    print("extracting absolute stargan_models")
    zip_ref.extractall('conditional_models/aligned_models/stargan_absolute_distance/')
os.remove("stargan_models_abs.zip")