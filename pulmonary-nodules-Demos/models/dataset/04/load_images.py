import pandas as pd
from PIL import Image
import requests
import numpy as np
from numpy import array
from io import StringIO, BytesIO
import urllib.request
import os


def save_image(_url, _filename, _index):
    with urllib.request.urlopen(_url) as url:
        f = BytesIO(url.read())
    img = Image.open(f).convert('RGB')
    img.save('train_images/' + _filename)
    print(_index)


if not os.path.exists(os.getcwd() + '/train_images'):
    os.makedirs(os.getcwd() + '/train_images')
data = pd.read_csv('dataset/book30-listing-train.csv', encoding="ISO-8859-1",
                   names=['ASIN', 'FILENAME', 'IMAGE_URL', 'TITLE', 'AUTHOR', 'CATEGORY_ID', 'CATEGORY'])
data["index"] = data.index
data = data.head()
data[["IMAGE_URL", "FILENAME", "index"]].apply(lambda x: save_image(*x), axis=1)
