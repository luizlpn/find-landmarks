from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import uvicorn

app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


LANDMARK_MAPPING = {
    # Íris - CORREÇÃO: trocado left/right
    "leftIris": 468,    # Íris esquerda (ponto central)
    "rightIris": 473,   # Íris direita (ponto central)
    
    # Cantos dos olhos 
    "leftLateralCanthus": 33,     # Canto externo olho esquerdo
    "leftMedialCanthus": 133,     # Canto interno olho esquerdo
    "rightLateralCanthus": 263,   # Canto externo olho direito
    "rightMedialCanthus": 362,    # Canto interno olho direito
    
    # Pálpebras - 
    "leftEyeUpper": 159,    # Ponto superior olho esquerdo
    "leftEyeLower": 145,    # Ponto inferior olho esquerdo
    "rightEyeUpper": 386,   # Ponto superior olho direito
    "rightEyeLower": 374,   # Ponto inferior olho direito
    
    # Sobrancelhas - 
    "leftEyebrow": 107,     # Ponto central sobrancelha esquerda
    "rightEyebrow": 336,    # Ponto central sobrancelha direita
    
    # Silhueta 
    "leftZygo": 58,      # Zigomo esquerdo
    "rightZygo": 288,    # Zigomo direito
    "leftGonial": 172,   # Ângulo mandibular esquerdo
    "rightGonial": 397,  # Ângulo mandibular direito
    "chinLeft": 200,     # Queixo lado esquerdo
    "chinTip": 152,      # Ponta do queixo
    "chinRight": 427,    # Queixo lado direito
    
    # Nariz 
    "noseBottom": 4,        # Ponta do nariz
    "leftNoseCorner": 129,  # Narina esquerda
    "rightNoseCorner": 358, # Narina direita
    
    # Lábios
    "leftCupidBow": 291,    # Arco de cupido esquerdo
    "rightCupidBow": 61,    # Arco de cupido direito
    "leftLipCorner": 61,    # Canto esquerdo lábios
    "rightLipCorner": 291,  # Canto direito lábios
    "upperLip": 0,          # Centro lábio superior
    "lipSeparation": 13,    # Centro separação lábios
    "lowerLip": 17          # Centro lábio inferior
}

@app.post("/detect-landmarks")
async def detect_landmarks(file: UploadFile = File(...)):
    try:
        # Lê a imagem
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
