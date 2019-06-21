import sys
import os
import mimetypes
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))

from classifiers import MQClassifier
from preprocessing import PCA

def get_image(subject, expression):
    image = Image.open("./YALE-A/%s.%s" % (subject, expression))
    image = image.resize((50, 50), Image.ANTIALIAS)
    return np.array(image)

faces = []
for _, _, files in os.walk("./YALE-A"):
    for filename in files:
        if filename.startswith('subject'):
            faces.append(filename)
faces = [face.split('.') for face in faces]

images = [[[subject, expression], get_image(subject, expression)] for subject, expression in faces]

print(images[0][1].shape)



