import cv2
import numpy as np
import base64

class Confidence:
    def __init__(self, id, value):
        self.pis = id
        self.value = value

class FaceRecognizer:
    def __init__(self):
        self.detectorFace = cv2.CascadeClassifier('app/cascade/haarcascade_frontalface_default.xml')
        self.reconhecedor = cv2.face.LBPHFaceRecognizer_create()
        self.reconhecedor.read("app/classifier/classificadorLBPH.yml")
        self.height, self.width = 220, 220

    def recognize_face(self, img_beat):
        decoded_data = base64.b64decode(img_beat)
        np_data = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_detect = self.detectorFace.detectMultiScale(
            image_gray,
            scaleFactor=1.5,
            minSize=(35, 35),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        confidences = []

        for (x, y, h, w) in face_detect:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            face_image = cv2.resize(image_gray[y:y+h, x:x+w], (self.width, self.height))
            
            id, confianca = self.reconhecedor.predict(face_image)

            confidence_obj = Confidence(id=id, value=round(confianca, 2))
            confidences.append(confidence_obj)

        # Se houver pelo menos uma detecção, retorne o dicionário
        if confidences:
            return {"pis": confidences[0].pis, "confidence": confidences[0].value}
        else:
            return {"pis": None, "confidence": None}