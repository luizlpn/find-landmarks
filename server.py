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

# Mapeamento 100% alinhado ao analysis.js
LANDMARK_MAPPING = {
    # Íris 
    "leftIris": {"source": "rightEyeIris", "index": 0},  # face.annotations.rightEyeIris[0]
    "rightIris": {"source": "leftEyeIris", "index": 0},  # face.annotations.leftEyeIris[0]
    
    # Cantos dos olhos (índices idênticos ao analysis.js)
    "leftLateralCanthus": {"source": "rightEyeLower1", "index": 0},  # face.annotations.rightEyeLower1[0]
    "leftMedialCanthus": {"source": "rightEyeLower1", "index": 7},   # face.annotations.rightEyeLower1[7]
    "rightLateralCanthus": {"source": "leftEyeLower1", "index": 0},  # face.annotations.leftEyeLower1[0]
    "rightMedialCanthus": {"source": "leftEyeLower1", "index": 7},   # face.annotations.leftEyeLower1[7]
    
    # Pálpebras 
    "leftEyeUpper": {"source": "rightEyeUpper0", "index": 4},  # face.annotations.rightEyeUpper0[4]
    "leftEyeLower": {"source": "rightEyeLower0", "index": 4},  # face.annotations.rightEyeLower0[4]
    "rightEyeUpper": {"source": "leftEyeUpper0", "index": 4},  # face.annotations.leftEyeUpper0[4]
    "rightEyeLower": {"source": "leftEyeLower0", "index": 4},  # face.annotations.leftEyeLower0[4]
    
    # Sobrancelhas 
    "leftEyebrow": {"source": "rightEyebrowUpper", "index": 6},  # face.annotations.rightEyebrowUpper[6]
    "rightEyebrow": {"source": "leftEyebrowUpper", "index": 6},  # face.annotations.leftEyebrowUpper[6]
    
    # Silhueta 
    "leftZygo": {"source": "silhouette", "index": 28},  # face.annotations.silhouette[28]
    "rightZygo": {"source": "silhouette", "index": 8},  # face.annotations.silhouette[8]
    "leftGonial": {"source": "silhouette", "index": 24},  # face.annotations.silhouette[24]
    "rightGonial": {"source": "silhouette", "index": 12},  # face.annotations.silhouette[12]
    "chinLeft": {"source": "silhouette", "index": 19},  # face.annotations.silhouette[19]
    "chinTip": {"source": "silhouette", "index": 18},  # face.annotations.silhouette[18]
    "chinRight": {"source": "silhouette", "index": 17},  # face.annotations.silhouette[17]
    
    # Nariz 
    "noseBottom": {"source": "noseBottom", "index": 0},  # face.annotations.noseBottom[0]
    "leftNoseCorner": {"source": "noseRightCorner", "index": 0},  # face.annotations.noseRightCorner[0]
    "rightNoseCorner": {"source": "noseLeftCorner", "index": 0},  # face.annotations.noseLeftCorner[0]
    
    # Lábios 
    "leftCupidBow": {"source": "lipsUpperOuter", "index": 4},  # face.annotations.lipsUpperOuter[4]
    "rightCupidBow": {"source": "lipsUpperOuter", "index": 6},  # face.annotations.lipsUpperOuter[6]
    "leftLipCorner": {"source": "lipsUpperOuter", "index": 0},  # face.annotations.lipsUpperOuter[0]
    "rightLipCorner": {"source": "lipsUpperOuter", "index": 10},  # face.annotations.lipsUpperOuter[10]
    "upperLip": {"source": "lipsUpperOuter", "index": 5},  # face.annotations.lipsUpperOuter[5]
    "lipSeparation": {"source": "lipsUpperInner", "index": 5},  # face.annotations.lipsUpperInner[5]
    "lowerLip": {"source": "lipsLowerOuter", "index": 4}  # face.annotations.lipsLowerOuter[4]
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
            
            for name, config in LANDMARK_MAPPING.items():
                landmark = face_landmarks[config["index"]]
                landmarks[name] = [landmark.x, landmark.y, landmark.z]  # Formato [x, y, z]
            
            return {"landmarks": landmarks}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)