import cv2
import os
import numpy as np
from deepface import DeepFace
import threading
import time
from flask import Flask, render_template_string, Response, jsonify

# Configurações
photobase_path = "photobase"
# model_name = "ArcFace"
# detector_backend = "yolov11n"
# threshold = 3
model_name = "Dlib"
detector_backend = "yolov11n"
threshold = 0.45

# Carregar modelo
print("Carregando modelo...")
model = DeepFace.build_model(model_name)
print("Modelo carregado!")

# Carregar embeddings
print("Carregando base de dados...")
photo_db = []
recognized_people = []
distance_by_person = {}
original_people = set()
for person in os.listdir(photobase_path):
    person_dir = os.path.join(photobase_path, person)
    if os.path.isdir(person_dir):
        original_people.add(person)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                embedding = DeepFace.represent(img_path=img_path, detector_backend=detector_backend,
                                               anti_spoofing = True, model_name=model_name, enforce_detection=False)[0]['embedding']
                photo_db.append((person, embedding))
            except Exception as e:
                print(f"[!] Erro ao processar {img_path}: {e}")
print("Base de dados carregada!")

# Inicializações
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
app = Flask(__name__)
video_capture = cv2.VideoCapture(0)

compare_event = threading.Event()
last_detected = ""
last_detection_time = 0

recognition_confirmed = ""

# Comparação
def compare_face(face_img_path):
    global last_detected, last_detection_time, recognition_counts, recognition_confirmed, photo_db, recognized_people, distance_by_person

    try:
        print("[*] Comparando rosto...")
        embedding = DeepFace.represent(img_path=face_img_path, detector_backend=detector_backend,
                                        anti_spoofing = True, model_name=model_name, enforce_detection=False)[0]['embedding']
        temp_counts = {}  # Resetar contagens para esta rodada
        recognition_counts = {}
        for name, known_embedding in photo_db:
            distance = np.linalg.norm(np.array(embedding) - np.array(known_embedding))
            print(f"-> Comparando com {name}: distância = {distance:.2f}")
            if distance < threshold:
                temp_counts[name] = temp_counts.get(name, 0) + 1
                recognition_counts[name] = recognition_counts.get(name, 0) + 1
                print(f"[✓] {name} reconhecido {recognition_counts[name]}x")

                if recognition_counts[name] >= 2 and recognition_confirmed != name:
                    print(f"[✔] Rosto confirmado: {name}")
                    recognition_confirmed = name
                    last_detected = name
                    last_detection_time = time.time()

                    if name not in recognized_people:
                        recognized_people.append(name)
                        distance_by_person[name] = round(distance, 2)
                        photo_db = [(n, emb) for (n, emb) in photo_db if n != name]
                        recognition_counts = {}  # Resetar após reconhecimento confirmado

                        # Salvar imagem no histórico
                        historico_path = "historico"
                        os.makedirs(historico_path, exist_ok=True)
                        save_path = os.path.join(historico_path, f"{name}.jpg")
                        cv2.imwrite(save_path, cv2.imread(face_img_path))

                    return
        print("[x] Rosto não reconhecido.")
    except Exception as e:
        print(f"[!] Erro ao comparar rosto: {e}")
    finally:
        compare_event.clear()

# Stream de vídeo
progress_start_time = 0

def gen():
    global progress_start_time

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if time.time() - last_detection_time < 3:
                elapsed = time.time() - last_detection_time
                progress = min(int((elapsed / 3.0) * 300), 300)
                cv2.putText(frame, f"{last_detected}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y + h + 10), (x + progress, y + h + 30), (255, 255, 0), -1)
                cv2.rectangle(frame, (x, y + h + 10), (x + 300, y + h + 30), (200, 200, 200), 2)

            if not compare_event.is_set():
                if face_crop is not None and face_crop.size > 0 and face_crop.shape[0] >= 150 and face_crop.shape[1] >= 150:
                    compare_event.set()
                    face_img_path = "temp_face.jpg"
                    if cv2.imwrite(face_img_path, face_crop):
                        threading.Thread(target=compare_face, args=(face_img_path,)).start()
                    else:
                        print("[!] Falha ao salvar face_crop.")
                        compare_event.clear()
                else:
                    print("[!] Face crop vazia ou inválida.")

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# Rotas Flask
@app.route('/')
def index():
    return render_template_string("""
        <html>
            <head>
                <title>Reconhecimento Facial</title>
                <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
                <script>
                    function refreshList() {
                        $.getJSON("/status", function(data) {
                            $('#present').html(data.recognized.map(p => `<li>${p.nome} (distância: ${p.distancia})</li>`).join(''));
                            $('#not-recognized').html(data.not_recognized.map(p => `<li>${p}</li>`).join(''));
                        });
                    }
                    setInterval(refreshList, 3000);
                </script>
            </head>
            <body>
                <h1>Reconhecimento Facial</h1>
                <img src="{{ url_for('video_feed') }}">
                <h2>Lista de Presença</h2>
                <ul id="present"></ul>
                <h2>Cadastrados</h2>
                <ul id="not-recognized"></ul>
            </body>
        </html>
    """)

@app.route('/status')
def status():
    not_recognized_people = list(original_people - set(recognized_people))
    recognized_with_distances = [{"nome": p, "distancia": distance_by_person.get(p, "?")} for p in recognized_people]
    return jsonify({
        'recognized': recognized_with_distances,
        'not_recognized': not_recognized_people
    })

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Rodar servidor
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5006, debug=False)
    finally:
        video_capture.release()
