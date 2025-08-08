from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import uvicorn
import base64
from typing import Dict

app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Landmarks principais que devem ir para o JSON
LANDMARK_MAPPING = {
    "pupil_left": 468,
    "pupil_right": 473,
    "outer_canthus_left": 33,
    "inner_canthus_left": 133,
    "inner_canthus_right": 362,
    "upper_eyelid_left": 159,
    "lower_eyelid_left": 145,
    "upper_eyelid_right": 386,
    "lower_eyelid_right": 374,
    "jaw_left": 234,
    "jaw_right": 454,
    "chin": 152,
    "nose_tip": 1,
    "earlobe_left": 127,       # aproximado
    "cheekbone_left": 127,
    "cheekbone_right": 356,
    "brow_midpoint": 9,
    # eye_center será calculado dinamicamente
    "chin_left": 234,
    "chin_right": 454,
    "chin_top": 152,
    "chin_bottom": 200,        # aproximado
    "zygion_left": 127,
    "zygion_right": 356,
    "nose_bridge": 168,
    "upper_lip": 13,
    "hairline_center": 10,     # estimado
    "left_eye_outer": 33,
    "right_eye_outer": 263,

    # Landmarks para desenho (não retornam no JSON final)
    "_right_canthus_outer": 263,
    "_nose_left_corner": 98,
    "_nose_right_corner": 327,
    "_left_lip_corner": 61,
    "_right_lip_corner": 291,
    "_lower_lip_center": 17
}

# Conexões para formar o desenho na imagem
LANDMARK_CONNECTIONS = [
    # Contorno do rosto
    ("jaw_left", "chin_left"),
    ("chin_left", "chin"),
    ("chin", "chin_right"),
    ("chin_right", "jaw_right"),

    # Linha do nariz
    ("nose_bridge", "nose_tip"),
    ("_nose_left_corner", "nose_tip"),
    ("_nose_right_corner", "nose_tip"),

    # Olho esquerdo
    ("outer_canthus_left", "upper_eyelid_left"),
    ("upper_eyelid_left", "inner_canthus_left"),
    ("inner_canthus_left", "lower_eyelid_left"),
    ("lower_eyelid_left", "outer_canthus_left"),

    # Olho direito
    ("inner_canthus_right", "upper_eyelid_right"),
    ("upper_eyelid_right", "_right_canthus_outer"),
    ("_right_canthus_outer", "lower_eyelid_right"),
    ("lower_eyelid_right", "inner_canthus_right"),

    # Boca
    ("_left_lip_corner", "upper_lip"),
    ("upper_lip", "_right_lip_corner"),
    ("_right_lip_corner", "_lower_lip_center"),
    ("_lower_lip_center", "_left_lip_corner")
]

# Cor única (#AD00FF em BGR)
PURPLE = (255, 0, 173)

def draw_wireframe(image: np.ndarray, landmarks: Dict):
    """Desenha as conexões entre os landmarks"""
    for (start_name, end_name) in LANDMARK_CONNECTIONS:
        if start_name in landmarks and end_name in landmarks:
            start_point = (
                int(landmarks[start_name]["pixel_coords"]["x"]),
                int(landmarks[start_name]["pixel_coords"]["y"])
            )
            end_point = (
                int(landmarks[end_name]["pixel_coords"]["x"]),
                int(landmarks[end_name]["pixel_coords"]["y"])
            )
            cv2.line(image, start_point, end_point, PURPLE, 1, cv2.LINE_AA)

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
        all_landmarks = {}  # Inclui também os de desenho

        # Extrai coordenadas dos pontos
        for name, index in LANDMARK_MAPPING.items():
            landmark = face_landmarks[index]
            all_landmarks[name] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "pixel_coords": {
                    "x": int(landmark.x * width),
                    "y": int(landmark.y * height)
                }
            }
            # Apenas os que não começam com "_" vão para o JSON
            if not name.startswith("_"):
                landmarks[name] = all_landmarks[name]

        # Calcula eye_center
        left_point = face_landmarks[159]
        right_point = face_landmarks[386]
        avg_x = (left_point.x + right_point.x) / 2
        avg_y = (left_point.y + right_point.y) / 2
        avg_z = (left_point.z + right_point.z) / 2
        eye_center_data = {
            "x": avg_x,
            "y": avg_y,
            "z": avg_z,
            "pixel_coords": {
                "x": int(avg_x * width),
                "y": int(avg_y * height)
            }
        }
        landmarks["eye_center"] = eye_center_data
        all_landmarks["eye_center"] = eye_center_data

        # Prepara imagem com wireframe e pontos
        annotated_image = image.copy()
        draw_wireframe(annotated_image, all_landmarks)

        for coords in all_landmarks.values():
            x, y = coords["pixel_coords"]["x"], coords["pixel_coords"]["y"]
            cv2.circle(annotated_image, (x, y), 5, PURPLE, -1)

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
