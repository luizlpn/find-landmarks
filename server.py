from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import uvicorn
import random

app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Mapeamento completo de landmarks
LANDMARK_MAPPING = {
    "leftIris": 468,
    "rightIris": 473,
    "leftLateralCanthus": 33,
    "leftMedialCanthus": 133,
    "rightLateralCanthus": 362,
    "rightMedialCanthus": 263,
    "leftEyeUpper": 159,
    "leftEyeLower": 145,
    "rightEyeUpper": 386,
    "rightEyeLower": 374,
    "leftEyebrow": 107,
    "rightEyebrow": 336,
    "leftZygo": 58,
    "rightZygo": 288,
    "leftGonial": 199,
    "rightGonial": 423,
    "chinLeft": 200,
    "chinTip": 152,
    "chinRight": 427,
    "noseBottom": 4,
    "leftNoseCorner": 129,
    "rightNoseCorner": 358,
    "leftCupidBow": 291,
    "rightCupidBow": 61,
    "leftLipCorner": 61,
    "rightLipCorner": 291,
    "upperLip": 13,
    "lipSeparation": 14,
    "lowerLip": 17
}

# Paleta de cores únicas para cada landmark
COLORS = {
    "leftIris": (255, 0, 0),        # Vermelho (BGR)
    "rightIris": (0, 255, 0),       # Verde
    "leftLateralCanthus": (255, 0, 255),  # Magenta
    "leftMedialCanthus": (0, 255, 255),   # Ciano
    "rightLateralCanthus": (255, 255, 0), # Amarelo
    "rightMedialCanthus": (0, 0, 255),    # Azul
    "leftEyeUpper": (128, 0, 128),  # Roxo
    "leftEyeLower": (0, 128, 128),  # Teal
    "rightEyeUpper": (128, 128, 0), # Oliva
    "rightEyeLower": (128, 0, 0),   # Marrom
    "leftEyebrow": (0, 128, 0),     # Verde escuro
    "rightEyebrow": (0, 0, 128),    # Azul marinho
    "leftZygo": (255, 165, 0),      # Laranja
    "rightZygo": (0, 165, 255),     # Azul claro
    "leftGonial": (128, 128, 128),  # Cinza
    "rightGonial": (64, 64, 64),    # Cinza escuro
    "chinLeft": (255, 192, 203),    # Rosa
    "chinTip": (0, 255, 127),       # Verde primavera
    "chinRight": (255, 215, 0),     # Ouro
    "noseBottom": (70, 130, 180),   # Azul aço
    "leftNoseCorner": (240, 128, 128), # Salmão
    "rightNoseCorner": (147, 112, 219), # Púrpura médio
    "leftCupidBow": (220, 20, 60),  # Carmesim
    "rightCupidBow": (95, 158, 160), # Azul cadete
    "leftLipCorner": (218, 165, 32), # Dourado
    "rightLipCorner": (50, 205, 50), # Verde lima
    "upperLip": (138, 43, 226),     # Violeta
    "lipSeparation": (255, 105, 180), # Rosa quente
    "lowerLip": (75, 0, 130)        # Índigo
}

# Formas para cada landmark
SHAPES = {
    "circle": lambda img, center, color, size: cv2.circle(img, center, size, color, -1),
    "square": lambda img, center, color, size: cv2.rectangle(
        img, 
        (center[0]-size, center[1]-size), 
        (center[0]+size, center[1]+size), 
        color, -1
    ),
    "triangle": lambda img, center, color, size: cv2.drawContours(
        img, 
        [np.array([
            (center[0], center[1]-size),
            (center[0]-size, center[1]+size),
            (center[0]+size, center[1]+size)
        ])], 
        0, color, -1
    ),
    "diamond": lambda img, center, color, size: cv2.drawContours(
        img, 
        [np.array([
            (center[0], center[1]-size),
            (center[0]-size, center[1]),
            (center[0], center[1]+size),
            (center[0]+size, center[1])
        ])], 
        0, color, -1
    )
}

# Atribuição de formas para cada landmark
SHAPE_ASSIGNMENTS = {
    "leftIris": "circle",
    "rightIris": "circle",
    "leftLateralCanthus": "square",
    "leftMedialCanthus": "square",
    "rightLateralCanthus": "square",
    "rightMedialCanthus": "square",
    "leftEyeUpper": "triangle",
    "leftEyeLower": "triangle",
    "rightEyeUpper": "triangle",
    "rightEyeLower": "triangle",
    "leftEyebrow": "diamond",
    "rightEyebrow": "diamond",
    "leftZygo": "circle",
    "rightZygo": "circle",
    "leftGonial": "diamond",
    "rightGonial": "diamond",
    "chinLeft": "triangle",
    "chinTip": "circle",
    "chinRight": "triangle",
    "noseBottom": "circle",
    "leftNoseCorner": "square",
    "rightNoseCorner": "square",
    "leftCupidBow": "triangle",
    "rightCupidBow": "triangle",
    "leftLipCorner": "diamond",
    "rightLipCorner": "diamond",
    "upperLip": "square",
    "lipSeparation": "circle",
    "lowerLip": "square"
}

@app.post("/visualize")
async def visualize_landmarks(file: UploadFile = File(...)):
    try:
        # Ler a imagem
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        height, width, _ = image.shape
        
        # Processar com MediaPipe
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks:
                raise HTTPException(status_code=400, detail="No face detected")
            
            # Desenhar os landmarks
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            for name, index in LANDMARK_MAPPING.items():
                landmark = face_landmarks[index]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                color = COLORS[name]
                shape_name = SHAPE_ASSIGNMENTS[name]
                shape_fn = SHAPES[shape_name]
                
                # Desenhar o landmark com cor e forma específicas
                shape_fn(image, (x, y), color, 6)
            
            # Converter para JPEG
            _, buffer = cv2.imencode('.jpg', image)
            return Response(content=buffer.tobytes(), media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
