from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import uvicorn
import base64
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
from io import BytesIO

app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Mapeamento atualizado com índices precisos
LANDMARK_MAPPING = {
    # Íris (correto)
    "leftIris": 468,    # Íris esquerda (ponto central)
    "rightIris": 473,   # Íris direita (ponto central)
    
    # Cantos dos olhos (ajustados para correspondência anatômica)
    "leftLateralCanthus": 33,     # Canto externo olho esquerdo
    "leftMedialCanthus": 133,     # Canto interno olho esquerdo
    "rightLateralCanthus": 362,   # Canto externo olho direito
    "rightMedialCanthus": 263,    # Canto interno olho direito
    
    # Pálpebras (correto)
    "leftEyeUpper": 159,    # Ponto superior olho esquerdo
    "leftEyeLower": 145,    # Ponto inferior olho esquerdo
    "rightEyeUpper": 386,   # Ponto superior olho direito
    "rightEyeLower": 374,   # Ponto inferior olho direito
    
    # Sobrancelhas (correto)
    "leftEyebrow": 107,     # Ponto central sobrancelha esquerda
    "rightEyebrow": 336,    # Ponto central sobrancelha direita
    
    # Silhueta (ajustes importantes)
    "leftZygo": 58,      # Zigomo esquerdo
    "rightZygo": 288,    # Zigomo direito
    "leftGonial": 199,   # Ângulo mandibular esquerdo (corrigido)
    "rightGonial": 423,  # Ângulo mandibular direito (corrigido)
    "chinLeft": 200,     # Queixo lado esquerdo
    "chinTip": 152,      # Ponta do queixo
    "chinRight": 427,    # Queixo lado direito
    
    # Nariz (correto)
    "noseBottom": 4,        # Ponta do nariz
    "leftNoseCorner": 129,  # Narina esquerda
    "rightNoseCorner": 358, # Narina direita
    
    # Lábios (ajustes críticos)
    "leftCupidBow": 291,    # Arco de cupido esquerdo
    "rightCupidBow": 61,    # Arco de cupido direito
    "leftLipCorner": 61,    # Canto esquerdo lábios
    "rightLipCorner": 291,  # Canto direito lábios
    "upperLip": 13,         # Centro lábio superior (corrigido)
    "lipSeparation": 14,    # Centro separação lábios (corrigido)
    "lowerLip": 17          # Centro lábio inferior
}

# Cores únicas para cada landmark (formato BGR)
LANDMARK_COLORS = {
    "leftIris": (255, 0, 0),        # Azul (BGR: 255,0,0)
    "rightIris": (0, 255, 0),       # Verde (BGR: 0,255,0)
    "leftLateralCanthus": (0, 0, 255),     # Vermelho (BGR: 0,0,255)
    "leftMedialCanthus": (255, 255, 0),    # Ciano (BGR: 255,255,0)
    "rightLateralCanthus": (0, 255, 255),  # Amarelo (BGR: 0,255,255)
    "rightMedialCanthus": (128, 0, 128),   # Roxo (BGR: 128,0,128)
    "leftEyeUpper": (0, 165, 255),         # Laranja (BGR: 0,165,255)
    "leftEyeLower": (203, 192, 255),       # Rosa (BGR: 203,192,255)
    "rightEyeUpper": (130, 0, 75),         # Marrom (BGR: 130,0,75)
    "rightEyeLower": (211, 0, 148),        # Magenta (BGR: 211,0,148)
    "leftEyebrow": (0, 128, 128),          # Teal (BGR: 0,128,128)
    "rightEyebrow": (128, 128, 0),         # Oliva (BGR: 128,128,0)
    "leftZygo": (0, 69, 255),              # Vermelho-escuro (BGR: 0,69,255)
    "rightZygo": (87, 139, 46),            # Verde-escuro (BGR: 87,139,46)
    "leftGonial": (128, 0, 0),             # Azul-marinho (BGR: 128,0,0)
    "rightGonial": (128, 128, 128),        # Cinza (BGR: 128,128,128)
    "chinLeft": (0, 140, 255),             # Dourado (BGR: 0,140,255)
    "chinTip": (226, 43, 138),             # Violeta (BGR: 226,43,138)
    "chinRight": (133, 21, 199),           # Azul-violeta (BGR: 133,21,199)
    "noseBottom": (42, 42, 165),           # Carmim (BGR: 42,42,165)
    "leftNoseCorner": (80, 127, 255),      # Coral (BGR: 80,127,255)
    "rightNoseCorner": (0, 215, 255),      # Ouro (BGR: 0,215,255)
    "leftCupidBow": (147, 20, 255),        # Orquídea (BGR: 147,20,255)
    "rightCupidBow": (112, 25, 25),        # Sépia (BGR: 112,25,25)
    "leftLipCorner": (47, 107, 85),        # Verde-azulado (BGR: 47,107,85)
    "rightLipCorner": (204, 209, 72),      # Verde-amarelado (BGR: 204,209,72)
    "upperLip": (50, 205, 154),            # Verde-marinho (BGR: 50,205,154)
    "lipSeparation": (92, 11, 227),        # Índigo (BGR: 92,11,227)
    "lowerLip": (180, 105, 255)            # Rosa-choque (BGR: 180,105,255)
}

@app.post("/detect-landmarks")
async def detect_landmarks(file: UploadFile = File(...)):
    try:
        # Lê a imagem
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        height, width, _ = image.shape
        
        # Processa com MediaPipe
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks:
                raise HTTPException(status_code=400, detail="No face detected")
            
            # Extrai os landmarks
            face_landmarks = results.multi_face_landmarks[0].landmark
            landmarks = {}
            
            for name, index in LANDMARK_MAPPING.items():
                landmark = face_landmarks[index]
                landmarks[name] = [landmark.x, landmark.y, landmark.z]
            
            return {"landmarks": landmarks}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize-landmarks")
async def visualize_landmarks(file: UploadFile = File(...)):
    try:
        # Lê a imagem
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        # Processa com MediaPipe
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                raise HTTPException(status_code=400, detail="No face detected")
            
            # Desenha os landmarks na imagem
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # Desenha todos os landmarks do rosto (pequenos pontos)
            for landmark in face_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(image, (x, y), 1, (100, 100, 100), -1)
            
            # Desenha os landmarks específicos com cores únicas
            for landmark_name, index in LANDMARK_MAPPING.items():
                landmark = face_landmarks[index]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                color = LANDMARK_COLORS[landmark_name]
                cv2.circle(image, (x, y), 5, color, -1)
            
            # Converte a imagem para base64 para retornar
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {"image": image_base64}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
