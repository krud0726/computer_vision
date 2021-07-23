# -*- coding: utf-8 -*-
"""deep_face_recognition_practice.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/106so3zW_IwNHU5M0LVVV_HLr2Y4Qh4K1

# Example1. Face Detection using Retina-Face
"""

!pip install retina-face

from retinaface import RetinaFace

from google.colab.patches import cv2_imshow

from google.colab import drive
drive.mount('/content/drive')

import cv2
import matplotlib.pyplot as plt

IMG_PATH = "/content/drive/MyDrive/people/iu/iu.jpg"

img = cv2.imread(IMG_PATH)

cv2_imshow(img)

obj = RetinaFace.detect_faces(IMG_PATH)

print(obj)

for key in obj.keys():
  identity = obj[key];
  print(identity)

  facial_areas = identity['facial_area']
  cv2.rectangle(img, (facial_areas[2], facial_areas[3]), (facial_areas[0], facial_areas[1]), (255,255,255,), 1)

plt.figure(figsize = (20, 20))
plt.imshow(img[:, :, ::-1])
plt.show()

"""# Example2. Deep Face Recognition"""

!pip install deepface

from deepface import DeepFace

DeepFace.verify(img1_path = "/content/drive/MyDrive/people/iu/iu.jpg", img2_path = "/content/drive/MyDrive/unpeople/bongseon.jpg", model_name = 'ArcFace', detector_backend= 'retinaface')

df = DeepFace.find(img_path = "/content/drive/MyDrive/people/iu/iu.jpg", db_path = "/content/drive/MyDrive/unpeople", model_name = 'ArcFace', detector_backend= 'retinaface')

"""# Example3. insight face"""

# insightface

RetinaFace.