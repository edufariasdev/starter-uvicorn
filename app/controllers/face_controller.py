# controller.py
import cv2
import os
from fastapi import HTTPException
import numpy as np
import base64
from app.use_cases import trainning
from app.use_cases.recognizer_lbph import FaceRecognizer

class FaceController:

    @staticmethod
    def process_face(body):
        try:
            eigenface = cv2.face.EigenFaceRecognizer_create()
            lbph = cv2.face.LBPHFaceRecognizer_create()

            cascPath = 'app/cascade/haarcascade_frontalface_default.xml'
            facePath = cv2.CascadeClassifier(cascPath)

            width, height = 220, 220
            decoded_data = base64.b64decode(body.origin)
            np_data = np.frombuffer(decoded_data, np.uint8)
            img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            print(np.average(gray))

            face_detect = facePath.detectMultiScale(
                gray,
                scaleFactor=1.5,
                minSize=(35, 35),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in face_detect:
                face_off = cv2.resize(gray[y:y + h, x:x + w], (width, height))
                cv2.imwrite('app/photo/person.' + str(body.pis) + '.' + str("1") + '.jpg', face_off)

            ids, faces = trainning.getImageWithId()

            eigenface.train(faces, ids)
            eigenface.write('app/classifier/classificadorEigen.yml')
            lbph.train(faces, ids)
            lbph.write('app/classifier/classificadorLBPH.yml')
            face_recognizer = FaceRecognizer()
            response = face_recognizer.recognize_face(body.image_beat)

            print('log de processamento', response)
            os.remove('app/photo/person.' + str(body.pis) + '.' + str("1") + '.jpg')
            return response
        
        except HTTPException as exc:
            # Captura a exceção HTTPException e a retorna como resposta de erro
            return {"error": exc.detail}, exc.status_code
