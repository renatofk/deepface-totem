import cv2
import os
import numpy as np
from flask import Flask, render_template_string, Response
from deepface import DeepFace
import threading
import time

# ========== CONFIGURAÇÕES ==========
base_path = "photobase"
model_name = "ArcFace"
embedding_interval = 5  # segundos entre comparações
video_width = 640
video_height = 480

# ========== FLASK ==========
app = Flask(__name__)

# ========== CARREGAR BASE DE EMBEDDINGS ==========
def carregar_base():
    embeddings_base = []
    for pessoa in os.listdir(base_path):
        pasta_pessoa = os.path.join(base_path, pessoa)
        if os.path.isdir(pasta_pessoa):
            for img_file in os.listdir(pasta_pessoa):
                img_path = os.path.join(pasta_pessoa, img_file)
                try:
                    emb = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=False)[0]['embedding']
                    embeddings_base.append((pessoa, emb))
                except:
                    continue
    return embeddings_base

embeddings_base = carregar_base()

# ========== FUNÇÃO DE COMPARAÇÃO ==========
def comparar_face(embedding_capturado):
    from scipy.spatial.distance import cosine
    min_dist = float('inf')
    identidade = "Desconhecido"
    for nome, emb_base in embeddings_base:
        dist = cosine(embedding_capturado, emb_base)
        if dist < 0.4 and dist < min_dist:
            min_dist = dist
            identidade = nome
    return identidade, min_dist

# ========== THREAD DE VERIFICAÇÃO ==========
last_embedding_time = 0
identificacao_atual = "Aguardando rosto..."

lock = threading.Lock()

def verificar_rosto(frame_clone):
    global last_embedding_time, identificacao_atual
    try:
        emb = DeepFace.represent(img_path=frame_clone, model_name=model_name, enforce_detection=False)[0]['embedding']
        identidade, dist = comparar_face(emb)
        with lock:
            identificacao_atual = f"{identidade} (dist={dist:.2f})"
    except Exception as e:
        print(f"Erro ao comparar rosto: {e}")
        with lock:
            identificacao_atual = "Erro ao identificar"
    last_embedding_time = time.time()

# ========== GERADOR DE FRAMES ==========
def gen():
    global last_embedding_time
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in rostos:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            current_time = time.time()
            if (current_time - last_embedding_time) > embedding_interval:
                face_img = frame[y:y + h, x:x + w].copy()
                face_img = cv2.resize(face_img, (112, 112))  # Tamanho esperado pelo ArcFace
                threading.Thread(target=verificar_rosto, args=(face_img,)).start()

        with lock:
            texto = identificacao_atual
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# ========== ROTAS ==========
@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head><title>Reconhecimento Facial</title></head>
        <body>
            <h2>Reconhecimento Facial em Tempo Real</h2>
            <img src="/video_feed">
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ========== EXECUÇÃO ==========
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
