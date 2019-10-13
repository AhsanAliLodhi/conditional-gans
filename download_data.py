import urllib.request
import zipfile
import os
import shutil

url = 'https://syncandshare.lrz.de/dl/fiYUvBh5ERiDYFDebLt9VVvr/data.zip'

print('Dlownloaing data')
urllib.request.urlretrieve(url, 'data.zip')
with zipfile.ZipFile('data.zip', 'r') as zip_ref:
    print("extracting data")
    zip_ref.extractall('data')
    
    
with zipfile.ZipFile('data/multicamera.zip', 'r') as zip_ref:
    print("extracting multicamera dataset")
    zip_ref.extractall('data/')
    
os.remove("data.zip")
os.remove("data/multicamera.zip")
