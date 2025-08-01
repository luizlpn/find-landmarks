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

# Cores diferentes para cada tipo de landmark
COLOR_MAPPING = {
    "Iris": (0, 255, 0),        # Verde
    "Olhos": (255, 0, 0),       # Azul
    "Sobrancelhas": (0, 0, 255),# Vermelho
    "Silhueta": (255, 255, 0),  # Ciano
    "Nariz": (255, 0, 255),     # Magenta
    "Lábios": (0, 255, 255),    # Amarelo
}

# Agrupamento de landmarks por categoria
LANDMARK_CATEGORIES = {
    "Iris": ["leftIris", "rightIris"],
    "Olhos": ["leftLateralCanthus", "leftMedialCanthus", "rightLateralCanthus", 
              "rightMedialCanthus", "leftEyeUpper", "leftEyeLower", 
              "rightEyeUpper", "rightEyeLower"],
    "Sobrancelhas": ["leftEyebrow", "rightEyebrow"],
    "Silhueta": ["leftZygo", "rightZygo", "leftGonial", "rightGonial", 
                 "chinLeft", "chinTip", "chinRight"],
    "Nariz": ["noseBottom", "leftNoseCorner", "rightNoseCorner"],
    "Lábios": ["leftCupidBow", "rightCupidBow", "leftLipCorner", 
               "rightLipCorner", "upperLip", "lipSeparation", "lowerLip"]
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
            
            # Desenha os landmarks específicos com cores diferentes
            for category, color in COLOR_MAPPING.items():
                for landmark_name in LANDMARK_CATEGORIES[category]:
                    if landmark_name in LANDMARK_MAPPING:
                        index = LANDMARK_MAPPING[landmark_name]
                        landmark = face_landmarks[index]
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        
                        # Desenha um círculo maior para o landmark
                        cv2.circle(image, (x, y), 5, color, -1)
                        
                        # Adiciona o nome do landmark
                        cv2.putText(image, landmark_name, (x+10, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Converte a imagem para base64 para retornar
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {"image": image_base64}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
