import os
import re
from PIL import Image
import numpy as np

__data_dir = "%s/YALE-A/" % os.path.dirname(__file__)

def load_data():
    faces = []
    for _, _, files in os.walk(__data_dir):
        for filename in files:
            if filename.startswith('subject'):
                faces.append(filename)
    faces = [face.split('.') for face in faces]
    data = [[subject, expression, __get_image(subject, expression)] for subject, expression in faces]
    return data

def get_images(data):
    images = [image for _, __, image in data]
    images = np.array(images)
    return images;

def get_subject_numbers(data):
    subjects =  [__get_subject_number(d) for d in data]
    subjects = np.array(subjects)
    return subjects

def __get_subject_number(data_entry):
    subject, _, __ = data_entry
    subject_number = re.sub('\D', '', subject)
    return int(subject_number)

def __get_image(subject, expression):
    image = Image.open("%s/%s.%s" % (__data_dir, subject, expression))
    image = image.resize((50, 50), Image.ANTIALIAS)
    return np.array(image).flatten()

