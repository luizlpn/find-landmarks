from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import uvicorn
import base64
from typing import Dict, List, Tuple

app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


LANDMARK_MAPPING = {
   
    "leftIris":           468,  # face.annotations.rightEyeIris[0]
    "rightIris":          473,  # face.annotations.leftEyeIris[0]

    # Cantos lateral/medial dos olhos (grupo LOWER1)
    "leftLateralCanthus": 33,   # face.annotations.rightEyeLower1[0]
    "leftMedialCanthus":  155,  # face.annotations.rightEyeLower1[7]
    "rightLateralCanthus":362,  # face.annotations.leftEyeLower1[0]
    "rightMedialCanthus": 249,  # face.annotations.leftEyeLower1[7]

    # Pálpebras (grupo UPPER0 e LOWER0)
    "leftEyeUpper":       158,  # face.annotations.rightEyeUpper0[4]
    "leftEyeLower":       145,  # face.annotations.rightEyeLower0[4]
    "rightEyeUpper":      385,  # face.annotations.leftEyeUpper0[4]
    "rightEyeLower":      374,  # face.annotations.leftEyeLower0[4]

    # Sobrancelhas (grupo EYEBROW_UPPER)
    "leftEyebrow":        296,  # face.annotations.rightEyebrowUpper[6]
    "rightEyebrow":       66,   # face.annotations.leftEyebrowUpper[6]

    # Zígomatico (silhouette) e gó­nio (silhouette)
    "leftZygo":           234,  # face.annotations.silhouette[28]
    "rightZygo":          454,  # face.annotations.silhouette[8]
    "leftGonial":         172,  # face.annotations.silhouette[24]
    "rightGonial":        397,  # face.annotations.silhouette[12]

    # Queixo (silhouette)
    "chinLeft":           148,  # face.annotations.silhouette[19]
    "chinTip":            152,  # face.annotations.silhouette[18]
    "chinRight":          377,  # face.annotations.silhouette[17]

    # Nariz
    "noseBottom":         2,    # face.annotations.noseBottom[0]
    "leftNoseCorner":     98,   # face.annotations.noseRightCorner[0]
    "rightNoseCorner":    327,  # face.annotations.noseLeftCorner[0]

    # Lábios (grupos UPPER_OUTER, UPPER_INNER e LOWER_OUTER)
    "leftCupidBow":       37,   # face.annotations.lipsUpperOuter[4]
    "lipSeparation":      14,   # face.annotations.lipsUpperInner[5]
    "rightCupidBow":      267,  # face.annotations.lipsUpperOuter[6]
    "leftLipCorner":      61,   # face.annotations.lipsUpperOuter[0]
    "rightLipCorner":     291,  # face.annotations.lipsUpperOuter[10]
    "lowerLip":           17,   # face.annotations.lipsLowerOuter[4]
    "upperLip":           0     # face.annotations.lipsUpperOuter[5]
}


PURPLE = (255, 0, 173)

def draw_wireframe(image: np.ndarray, landmarks: Dict, width: int, height: int):
    """Desenha as conexões entre os landmarks no estilo MediaPipe (todos roxos)"""
    # Mapeamento de conexões semânticas (similar ao MediaPipe)
    connections = [
        # Contorno do rosto (face oval)
        ("leftZygo", "leftGonial"),
        ("leftGonial", "chinLeft"),
        ("chinLeft", "chinTip"),
        ("chinTip", "chinRight"),
        ("chinRight", "rightGonial"),
        ("rightGonial", "rightZygo"),
        
        # Lábios
        ("leftLipCorner", "leftCupidBow"),
        ("leftCupidBow", "lipSeparation"),
        ("lipSeparation", "rightCupidBow"),
        ("rightCupidBow", "rightLipCorner"),
        ("rightLipCorner", "lowerLip"),
        ("lowerLip", "leftLipCorner"),
        
        # Olho direito
        ("rightLateralCanthus", "rightEyeUpper"),
        ("rightEyeUpper", "rightMedialCanthus"),
        ("rightMedialCanthus", "rightEyeLower"),
        ("rightEyeLower", "rightLateralCanthus"),
        
        # Sobrancelha direita
        ("rightLateralCanthus", "rightEyebrow"),
        ("rightEyebrow", "rightMedialCanthus"),
        
        # Olho esquerdo
        ("leftLateralCanthus", "leftEyeUpper"),
        ("leftEyeUpper", "leftMedialCanthus"),
        ("leftMedialCanthus", "leftEyeLower"),
        ("leftEyeLower", "leftLateralCanthus"),
        
        # Sobrancelha esquerda
        ("leftLateralCanthus", "leftEyebrow"),
        ("leftEyebrow", "leftMedialCanthus"),
        
        # Tesselation (malha facial)
        ("leftZygo", "leftEyebrow"),
        ("leftEyebrow", "leftMedialCanthus"),
        ("leftMedialCanthus", "leftNoseCorner"),
        ("leftNoseCorner", "noseBottom"),
        ("noseBottom", "rightNoseCorner"),
        ("rightNoseCorner", "rightMedialCanthus"),
        ("rightMedialCanthus", "rightEyebrow"),
        ("rightEyebrow", "rightZygo"),
        ("leftNoseCorner", "leftCupidBow"),
        ("rightNoseCorner", "rightCupidBow")
    ]

    # Desenha todas as conexões na cor roxa
    for (start_name, end_name) in connections:
        try:
            start_point = (
                int(landmarks[start_name]["pixel_coords"]["x"]),
                int(landmarks[start_name]["pixel_coords"]["y"])
            )
            end_point = (
                int(landmarks[end_name]["pixel_coords"]["x"]),
                int(landmarks[end_name]["pixel_coords"]["y"])
            )
            cv2.line(image, start_point, end_point, PURPLE, 1, cv2.LINE_AA)
        except KeyError:
            continue  # Ignora conexões com landmarks não encontrados

def process_image(image: np.ndarray) -> Dict:
    height, width, _ = image.shape

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            raise HTTPException(status_code=400, detail="No face detected")

        face_landmarks = results.multi_face_landmarks[0].landmark
        landmarks = {}

        # Extrai coordenadas
        for name, index in LANDMARK_MAPPING.items():
            landmark = face_landmarks[index]
            landmarks[name] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "pixel_coords": {
                    "x": int(landmark.x * width),
                    "y": int(landmark.y * height)
                }
            }

        # Prepara imagem com anotações
        annotated_image = image.copy()

        # 1. Desenha o wireframe primeiro (para ficar atrás dos pontos)
        draw_wireframe(annotated_image, landmarks, width, height)

        # 2. Desenha os pontos principais
        for name, coords in landmarks.items():
            x, y = coords["pixel_coords"]["x"], coords["pixel_coords"]["y"]
            cv2.circle(annotated_image, (x, y), 5, PURPLE, -1)
            cv2.putText(annotated_image, name, (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, PURPLE, 1)

        # Converte para base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        return {
            "landmarks": landmarks,
            "annotated_image": base64.b64encode(buffer).decode('utf-8'),
            "image_size": {"width": width, "height": height}
        }

@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    try:
        image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        return {"success": True, "data": process_image(image)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
