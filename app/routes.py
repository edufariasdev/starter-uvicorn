from fastapi import APIRouter
from pydantic import BaseModel, ValidationError, validator
from app.controllers.face_controller import FaceController
import base64

router = APIRouter()

def validate_base64_image(value: str, field_name: str):
    decoded_data = base64.b64decode(value)
    if not decoded_data:
        raise ValidationError(f'Você deve fornecer uma imagem base64 válida para o campo {field_name}!')

def validate_pis(value: str):
    if not value:
        raise ValidationError('Você deve fornecer um PIS para identificação!')

class FaceBody(BaseModel):
    origin: str
    image_beat: str
    pis: int

    @validator("origin")
    def validator_origin(cls, value: str):
        validate_base64_image(value, "origin")
        return value

    @validator("image_beat")
    def validator_image_beat(cls, value: str):
        validate_base64_image(value, "image_beat")
        return value
    
    @validator("pis")
    def validator_pis(cls, value: str):
        validate_pis(value)
        return value

@router.post("/v1/face")
async def face_root(body: FaceBody):
    return FaceController.process_face(body)