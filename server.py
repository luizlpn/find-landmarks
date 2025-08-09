# server.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import uvicorn
import base64
from typing import Dict, List, Union

app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ---------------------------
# LANDMARK MAPPING (os que vão no JSON)
# ---------------------------
LANDMARK_MAPPING: Dict[str, int] = {
    "leftIris": 468,
    "rightIris": 473,
    "outerCanthusLeft": 33,
    "leftMedialCanthus": 155,
    "rightLateralCanthus": 362,
    "rightMedialCanthus": 249,
    "leftEyeUpper": 158,
    "leftEyeLower": 145,
    "rightEyeUpper": 385,
    "rightEyeLower": 374,
    "leftEyebrow": 296,
    "rightEyebrow": 66,
    "leftZygo": 234,
    "rightZygo": 454,
    "leftGonial": 172,
    "rightGonial": 397,
    "chinLeft": 148,
    "chinTip": 152,
    "chinRight": 377,
    "noseBottom": 2,
    "leftNoseCorner": 98,
    "rightNoseCorner": 327,
    "upperLip": 0
}

# ---------------------------
# GRUPOS DE PONTOS SOMENTE PARA DESENHO (não entram no JSON)
# ---------------------------
# listas baseadas nas constantes comuns do MediaPipe (índices da mesh)
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
LEFT_EYE = [33,7,163,144,145,153,154,155,133]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263]
LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291]
NOSE_BRIDGE = [6,197,195,5,4]  # linha no dorso do nariz

# Cor do wireframe e pontos (BGR)
WIRE_COLOR = (0, 215, 255)      # amarelo-dourado para linhas
POINT_COLOR = (255, 0, 173)     # magenta/roxo para pontos (mantive tom próximo ao anterior)
LINE_THICKNESS = 1
LINE_THICKNESS_BOLD = 2
POINT_RADIUS = 4

# ---------------------------
# Utilitários
# ---------------------------
def to_pixel_coords(lm, width: int, height: int):
    return int(lm.x * width), int(lm.y * height)

def distance3D(p1, p2):
    return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)

def draw_polyline(image: np.ndarray, pts: List[tuple], closed: bool, color, thickness=1):
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        cv2.line(image, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)
    if closed:
        cv2.line(image, pts[-1], pts[0], color, thickness, cv2.LINE_AA)

# ---------------------------
# Processamento da imagem
# ---------------------------
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

        # dicionários para retorno (landmarks -> apenas os mapeados) e para desenho (todos)
        landmarks: Dict[str, Dict] = {}
        draw_points: Dict[str, List[Dict]] = {}

        # Extrai coordenadas dos pontos que devem ser enviados no JSON
        for name, idx in LANDMARK_MAPPING.items():
            lm = face_landmarks[idx]
            px = int(lm.x * width)
            py = int(lm.y * height)
            landmarks[name] = {
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "pixel_coords": {"x": px, "y": py}
            }

        # calcula eye_center (média entre pálpebras superiores conforme antes)
        try:
            left_pt = face_landmarks[159]
            right_pt = face_landmarks[386]
            avg_x = (left_pt.x + right_pt.x) / 2.0
            avg_y = (left_pt.y + right_pt.y) / 2.0
            avg_z = (left_pt.z + right_pt.z) / 2.0
            landmarks["eyeCenter"] = {
                "x": float(avg_x),
                "y": float(avg_y),
                "z": float(avg_z),
                "pixel_coords": {"x": int(avg_x * width), "y": int(avg_y * height)}
            }
        except Exception:
            # não interrompe; apenas não adiciona se der pau
            pass

        # -----------------------
        # Monta a imagem anotada (wireframe + pontos)
        # -----------------------
        annotated_image = image.copy()

        # 1) desenhar contorno do rosto (face oval)
        face_pts = []
        for idx in FACE_OVAL:
            lm = face_landmarks[idx]
            face_pts.append((int(lm.x * width), int(lm.y * height)))
        draw_polyline(annotated_image, face_pts, closed=True, color=WIRE_COLOR, thickness=LINE_THICKNESS_BOLD)

        # 2) olhos (desenhar left + right)
        left_eye_pts = []
        for idx in LEFT_EYE:
            lm = face_landmarks[idx]
            left_eye_pts.append((int(lm.x * width), int(lm.y * height)))
        draw_polyline(annotated_image, left_eye_pts, closed=True, color=WIRE_COLOR, thickness=LINE_THICKNESS)

        right_eye_pts = []
        for idx in RIGHT_EYE:
            lm = face_landmarks[idx]
            right_eye_pts.append((int(lm.x * width), int(lm.y * height)))
        draw_polyline(annotated_image, right_eye_pts, closed=True, color=WIRE_COLOR, thickness=LINE_THICKNESS)

        # 3) nariz (linha no dorso + base)
        nose_pts = []
        for idx in NOSE_BRIDGE:
            lm = face_landmarks[idx]
            nose_pts.append((int(lm.x * width), int(lm.y * height)))
        draw_polyline(annotated_image, nose_pts, closed=False, color=WIRE_COLOR, thickness=LINE_THICKNESS)

        # 4) boca (outer)
        lips_pts = []
        for idx in LIPS_OUTER:
            lm = face_landmarks[idx]
            lips_pts.append((int(lm.x * width), int(lm.y * height)))
        draw_polyline(annotated_image, lips_pts, closed=True, color=WIRE_COLOR, thickness=LINE_THICKNESS)

        # 5) desenhar linhas finas internas para reforçar "look tecnológico"
        #    - pequenos segmentos conectando centros dos olhos ao eyeCenter
        try:
            ec = landmarks["eyeCenter"]["pixel_coords"]
            lI = landmarks["leftIris"]["pixel_coords"]
            rI = landmarks["rightIris"]["pixel_coords"]
            cv2.line(annotated_image, (lI["x"], lI["y"]), (ec["x"], ec["y"]), WIRE_COLOR, 1, cv2.LINE_AA)
            cv2.line(annotated_image, (rI["x"], rI["y"]), (ec["x"], ec["y"]), WIRE_COLOR, 1, cv2.LINE_AA)
        except Exception:
            pass

        # 6) desenhar círculos nos landmarks principais (os que aparecem no JSON)
        for name, val in landmarks.items():
            try:
                px = val["pixel_coords"]["x"]
                py = val["pixel_coords"]["y"]
                cv2.circle(annotated_image, (px, py), POINT_RADIUS, POINT_COLOR, -1, cv2.LINE_AA)
            except Exception:
                continue

        # 7) leve borrão / efeito de brilho sutil (opcional) - desenhei uma borda sutil em volta do rosto
        # desenhar uma linha externa mais fina para dar sensação "scanner"
        face_outline_thin = []
        for idx in FACE_OVAL:
            lm = face_landmarks[idx]
            face_outline_thin.append((int(lm.x * width), int(lm.y * height)))
        draw_polyline(annotated_image, face_outline_thin, closed=True, color=(200,200,200), thickness=1)

        # Converte para base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        encoded_img = base64.b64encode(buffer).decode('utf-8')

        return {
            "landmarks": landmarks,
            "annotated_image": encoded_img,
            "image_size": {"width": width, "height": height}
        }

# Endpoint (mantido exatamente)
@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    try:
        image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        return {"success": True, "data": process_image(image)}
    except HTTPException as he:
        # repassa erro do FaceMesh (ex: no face)
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Mantive porta 8000 conforme solicitado
    uvicorn.run(app, host="0.0.0.0", port=8000)
