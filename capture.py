from flask import Flask, render_template, request, Response, url_for, redirect, jsonify
import cv2
import os
import numpy as np
import time
from threading import Thread
# import pyttsx3
from playsound import playsound
import threading
import queue
import base64
from deepface import DeepFace
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

model_name = "Dlib"

app = Flask(__name__)
# engine = pyttsx3.init()
# engine.setProperty('voice', 'brazil')
# engine.setProperty('rate', 120)

# Variáveis globais
student_name = ""
student_id = ""
foto_count = 0
total_fotos = 8
captura_finalizada = False
user_path = ""
ultima_foto = None
inicio_rosto_detectado = None
ultimo_tempo_captura = 0
intervalo_entre_fotos = 2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
threads = []
mensagem = ""
mensagem_cor = (255, 255, 255)

# Acesso do Zilliz Cloud:
zilliz_uri = "https://in03-729fdb949596dbe.serverless.gcp-us-west1.cloud.zilliz.com"
zilliz_token = "de6bc4285f95703df6b754e1440425ce33b76c28de022528b543936d7115dc36145935a8a7cabfada69a2d3ff0f37de5179a6401"

connections.connect(
    alias="default",
    uri=zilliz_uri,
    token=zilliz_token
    #secure=True  # Habilita TLS
)
print("✅ Conectado ao Zilliz Cloud!")

# Definição do esquema da coleção
collection_name = "student_embeddings"
# # Drop a collection if exists
# if collection_name in [c.name for c in Collection.list()]:
#     collection = Collection(name=collection_name)
#     collection.drop()
#     print(f"❌ Collection {collection_name} removida.")

# Crie a nova com a dimensão correta
fields = [
    FieldSchema(name="embedding_id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # ID único por embedding
    FieldSchema(name="school", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="student_id", dtype=DataType.INT64),
    FieldSchema(name="student_name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)  
]
schema = CollectionSchema(fields, description="Embeddings de estudantes")

try:
    collection = Collection(name=collection_name)
    print("Collection já existe.")
except Exception:
    collection = Collection(name=collection_name, schema=schema)
    print("Collection criada.")



# class FalaThreadSegura:
#     def __init__(self):
#         self.engine = pyttsx3.init()
#         self.engine.setProperty('rate', 120)
#         self.engine.setProperty('voice', self._get_voice('brazil'))
#         self.fila = queue.Queue()
#         self.thread = threading.Thread(target=self._executor, daemon=True)
#         self.thread.start()

#     def _get_voice(self, language_keyword):
#         for voice in self.engine.getProperty('voices'):
#             if language_keyword.lower() in voice.name.lower() or language_keyword.lower() in voice.id.lower():
#                 return voice.id
#         return self.engine.getProperty('voice')

#     def _executor(self):
#         while True:
#             texto = self.fila.get()
#             if texto is None:
#                 break
#             self.engine.say(texto)
#             self.engine.runAndWait()

#     def falar(self, texto):
#         self.fila.put(texto)

#     def parar(self):
#         self.fila.put(None)
#         self.thread.join()

# # ========== Uso ==========
# fala = FalaThreadSegura()

# def falar(texto):
#     fala.falar(texto)

def salvar_foto_em_thread(img, path, index):
    cv2.imwrite(path, img)
    playsound("camera-click.mp3")
    # falar(f"Foto salva.")

def esta_clara(img, limiar=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > limiar

def esta_nitida(img, limiar=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > limiar

def contem_rosto(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None

    # Pega o maior rosto (útil se houver mais de um)
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face_crop = img[y:y+h, x:x+w]
    return face_crop

def desenhar_barra_progresso(frame, progresso, total):
    largura = 300
    altura = 20
    x, y = 10, 420
    preenchimento = int((progresso / total) * largura)
    cv2.rectangle(frame, (x, y), (x + largura, y + altura), (255, 255, 255), 2)
    cv2.rectangle(frame, (x, y), (x + preenchimento, y + altura), (0, 255, 0), -1)

def generate_frames(student_id, student_name):
    global ultima_foto, inicio_rosto_detectado, ultimo_tempo_captura, mensagem, mensagem_cor, captura_finalizada

    captura_finalizada = False
    foto_count = 0
    user_path = os.path.join("photobase", student_name)
    os.makedirs(user_path, exist_ok=True)

    if captura_finalizada:
        texto = "Captura finalizada"
        cor = (0, 255, 0)

    else:
        texto = "Aguardando rosto..."
        cor = (0, 255, 255)
    
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        agora = time.time()    

        if not captura_finalizada:
            face_crop = contem_rosto(frame)
            
            if face_crop is not None:
                if inicio_rosto_detectado is None:
                    inicio_rosto_detectado = agora

                tempo_com_rosto = agora - inicio_rosto_detectado

                if tempo_com_rosto >= intervalo_entre_fotos and agora - ultimo_tempo_captura >= intervalo_entre_fotos:
                    if not esta_clara(face_crop):
                        mensagem = "[X] Imagem escura. Melhore a claridade"
                        mensagem_cor = (0, 0, 255)
                        # falar("Imagem escura")
                    elif not esta_nitida(face_crop):
                        mensagem = "[X] Imagem borrada. Mantenha o rosto firme e repita a foto."
                        mensagem_cor = (0, 0, 255)
                        # falar("Imagem borrada")
                    else:
                        foto_path = os.path.join(user_path, f"{str(student_id)}_{foto_count}.jpg")
                        thread = Thread(target=salvar_foto_em_thread, args=(face_crop.copy(), foto_path, foto_count))
                        thread.start()
                        threads.append(thread)
                        mensagem = f"Foto {foto_count + 1}/{total_fotos} salva!"
                        mensagem_cor = (0, 255, 0)

                        ultima_foto = face_crop.copy()
                        foto_count += 1
                        ultimo_tempo_captura = agora
                        if foto_count >= total_fotos:
                            captura_finalizada = True
                            texto = "Captura finalizada"
                            cor = (0, 255, 0)

                    inicio_rosto_detectado = None
                else:
                    texto = f"Captura em {max(0, int(intervalo_entre_fotos - tempo_com_rosto))} segundos..."
                    cor = (0, 255, 0)
            else:
                inicio_rosto_detectado = None

            desenhar_barra_progresso(frame, min(intervalo_entre_fotos, agora - inicio_rosto_detectado) if inicio_rosto_detectado else 0, intervalo_entre_fotos)

        # else:
        #    falar("Captura finalizada.")
        #    fala.parar()

        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
        cv2.putText(frame, f"Fotos: {foto_count}/{total_fotos}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if mensagem:
            cv2.putText(frame, mensagem, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mensagem_cor, 2)

        if ultima_foto is not None and not captura_finalizada:
            thumb = cv2.resize(ultima_foto, (100, 100))
            h, w, _ = thumb.shape
            frame[-h-10:-10, -w-10:-10] = thumb

        if captura_finalizada:
            time_to_stop = time.time()
            # Sai após 3 segundos
            while True:
                 if time.time() - time_to_stop > 3:
                    cap.release()
                    break # Encerra o loop do vídeo
 
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

dados_estudante = {}


def get_embedding(image):
    try:
        embedding = DeepFace.represent(image, model_name=model_name, enforce_detection=False)[0]['embedding']
        return embedding
    except Exception as e:
                    print(f"[!] Erro ao processar imagem: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    global dados_estudante
    if request.method == 'GET':
        student_name = request.form['student_name']
        student_id = request.form['student_id']
        dados_estudante = {'student_name': student_name, 'student_id': student_id}
        return redirect(url_for('camera'))
    return render_template('index.html')

@app.route('/camera')
def camera():
    student_name = request.args.get('student_name')
    student_id = request.args.get('student_id')

    if not student_name or not student_id:
        return redirect(url_for('index'))

    return render_template('camera.html', student_name=student_name, student_id=student_id)

@app.route('/video_feed')
def video_feed():
    student_name = request.args.get('student_name')
    student_id = request.args.get('student_id')
    return Response(generate_frames(student_id, student_name), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/status')
def status():
    return jsonify({'finalizado': captura_finalizada})

@app.route('/update_embeddings', methods=['POST'])
def update_embeddings():
    data = request.get_json()
    student_id = data['student_id']

    collection = Collection(collection_name)
    
    # Verifica se o student_id já existe para apagar os embeddings antigos
    count = collection.delete(f"student_id == {student_id}")
    print(f"Deletados {count} embeddings antigos do estudante {student_id}.")
    
    return jsonify({"status": "success", "message": "Embeddings removidos com sucesso."})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    image_data = data['image']
    student_id = data['student_id']
    student_name = data['student_name']
    position = data['position']
    print(f"Recebendo imagem para o aluno {student_name} com ID {student_id} e foto {position}")
    # Cria o diretório se não existir
    folder_path = os.path.join("photobase", student_name)
    os.makedirs(folder_path, exist_ok=True)

    # Remove o prefixo data:image/jpeg;base64,
    image_data = image_data.split(',')[1]

    # Decodifica a imagem
    img_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # (Opcional) Converter para RGB se o modelo precisar
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Inverte a imagem horizontalmente
    flipped_frame = cv2.flip(frame_rgb, 1)

    # Extrair o embedding da imagem (use seu modelo preferido)
    embedding = get_embedding(frame_rgb)
    embedding_flipped = get_embedding(flipped_frame)

    embeddings = [embedding, embedding_flipped]

    # # Printar o embedding no console
    # print("Embedding extraído:", embedding)

    # Salva o embedding no banco de dados
    # Inserir no Milvus/Zilliz

    data_to_insert = [
        {
            "school": "Unidade A",  # Nome da escola
            "student_id": int(student_id),
            "student_name": student_name,
            "embedding": emb
        }
        for emb in embeddings
    ]
    collection.insert(data_to_insert)
    print(f"✅ Embedding inserido para {student_name} (ID {student_id})")


    # Aqui você pode salvar ou processar a imagem (ex: detecção facial)
    foto_path = os.path.join(folder_path, f"{str(student_id)}_{position}.jpg")

    thread = Thread(target=salvar_foto_em_thread, args=(frame.copy(), foto_path, foto_count))
    thread.start()
    threads.append(thread)

    return jsonify({"status": "success", "message": "Imagem recebida e salva com sucesso."})

if __name__ == '__main__':

    # app.run(host='0.0.0.0',
    #         ssl_context='adhoc', 
    #         port=5003,
    #         debug=False)
    app.run(host='localhost', port=5003, debug=False)
   
   