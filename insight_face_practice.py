# -*- coding: utf-8 -*-
"""insight_face_practice.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iruuJHjGSKobnWAZlHM59UhVXf6bBnN1
"""

!pip install insightface
!pip install onnxruntime
!pip install mxnet

import insightface
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import sys

insightface.__version__

app = FaceAnalysis(allowed_modules = ['detection'])
app.prepare(ctx_id = 0, det_size=(640,640))

insightface.app.FaceAnalysis(name=)



app = FaceAnalysis(allowed_modules = ['detection'])
app.prepare(ctx_id = 0, det_size=(640,640))

detector = insightface.model_zoo.get_model('RetinaFace')

img = ins_get_image('t1')
faces = app.get(img)

rimg = app.draw_on(img, faces)
print(faces)
cv2.imwrite("./t1_output.jpg", rimg)

