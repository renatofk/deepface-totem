import cv2
import os
import numpy as np
import time
import pyttsx3
from threading import Thread
from playsound import playsound

# =============== Inicializações ===============
engine = pyttsx3.init()
engine.setProperty('voice', 'brazil')
engine.setProperty('rate', 120)  # Velocidade da fala

def falar(texto):
    engine.say(texto)
    engine.runAndWait()

def salvar_foto_em_thread(img, path, index):
    cv2.imwrite(path, img)
    playsound("camera-click.mp3")
    falar(f"Foto {index + 1} salva.")

# =============== Verificações de imagem ===============
def esta_clara(img, limiar=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > limiar

def esta_nitida(img, limiar=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > limiar

def contem_rosto(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

def desenhar_barra_progresso(frame, progresso, total):
    largura = 300
    altura = 20
    x, y = 10, 420
    preenchimento = int((progresso / total) * largura)
    cv2.rectangle(frame, (x, y), (x + largura, y + altura), (255, 255, 255), 2)
    cv2.rectangle(frame, (x, y), (x + preenchimento, y + altura), (0, 255, 0), -1)

# =============== Caminhos e setup ===============
nome = input("Digite seu nome: ").strip()
base_path = "photobase"
user_path = os.path.join(base_path, nome)
os.makedirs(user_path, exist_ok=True)

falar(f"Iniciando captura para {nome}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

foto_count = 0
total_fotos = 8

ultimo_tempo_captura = 0
inicio_rosto_detectado = None
intervalo_entre_fotos = 5  # segundos

mensagem = ""
mensagem_cor = (255, 255, 255)
ultima_foto = None
threads = []  # Lista para armazenar threads

# =============== Loop principal ===============
while foto_count < total_fotos:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a webcam.")
        break

    rostos = contem_rosto(frame, face_cascade)
    agora = time.time()

    if len(rostos) > 0:
        if inicio_rosto_detectado is None:
            inicio_rosto_detectado = agora

        tempo_com_rosto = agora - inicio_rosto_detectado

        if tempo_com_rosto >= intervalo_entre_fotos and agora - ultimo_tempo_captura >= intervalo_entre_fotos:
            if not esta_clara(frame):
                mensagem = "[X] Imagem escura. Melhore a claridade"
                mensagem_cor = (0, 0, 255)
                falar("Imagem escura. Tente novamente.")
            elif not esta_nitida(frame):
                mensagem = "[X] Imagem borrada. Mantenha o rosto firme."
                mensagem_cor = (0, 0, 255)
                falar("Imagem borrada. Repita a foto.")
            else:
                foto_path = os.path.join(user_path, f"{foto_count}.jpg")
                thread = Thread(target=salvar_foto_em_thread, args=(frame.copy(), foto_path, foto_count))
                thread.start()
                threads.append(thread)
                mensagem = f"[✓] Foto {foto_count + 1}/{total_fotos} salva!"
                mensagem_cor = (0, 255, 0)
                ultima_foto = frame.copy()
                foto_count += 1
                ultimo_tempo_captura = agora

            inicio_rosto_detectado = None  # Reiniciar para próxima sequência
        else:
            texto = f"Captura em {max(0, int(intervalo_entre_fotos - tempo_com_rosto))} segundos..."
            cor = (0, 255, 0)
    else:
        inicio_rosto_detectado = None
        texto = "Aguardando rosto..."
        cor = (0, 255, 255)

    # Interface visual
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
    cv2.putText(frame, f"Fotos aceitas: {foto_count}/{total_fotos}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if mensagem:
        cv2.putText(frame, mensagem, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mensagem_cor, 2)

    # Desenhar barra sincronizada com tempo de rosto detectado
    if inicio_rosto_detectado is not None:
        progresso = min(intervalo_entre_fotos, agora - inicio_rosto_detectado)
        desenhar_barra_progresso(frame, progresso, intervalo_entre_fotos)

    # Mostrar miniatura da última foto
    if ultima_foto is not None:
        thumb = cv2.resize(ultima_foto, (100, 100))
        h, w, _ = thumb.shape
        frame[-h-10:-10, -w-10:-10] = thumb

    cv2.imshow(f"Captura para {nome}", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =============== Encerramento ===============
cap.release()
cv2.destroyAllWindows()

# Aguardar término das threads
for thread in threads:
    thread.join()

falar("Captura finalizada.")
print(f"Captura finalizada. {foto_count} fotos salvas em {user_path}")
