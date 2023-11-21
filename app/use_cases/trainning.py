import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImageWithId():

    pathsImages = [os.path.join('app/photo', f) for f in os.listdir('app/photo')]
    faces = []
    ids = []

    for pathImage in pathsImages:
        imageFace = cv2.cvtColor(cv2.imread(pathImage), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(pathImage)[-1].split('.')[1])

        ids.append(id)
        faces.append(imageFace)

        cv2.waitKey(10)
    return np.array(ids), faces
