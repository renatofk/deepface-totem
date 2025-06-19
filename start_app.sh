#!/bin/bash

PID_FILE="/home/user/Dev/tmp/app_python.pid"

source /home/user/Dev/deepface-totem/totem/bin/activate

# Se o processo já estiver rodando, sai
if [ -f "$PID_FILE" ] && ps -p $(cat $PID_FILE) > /dev/null; then
    echo "O servidor já está rodando."
else
    # Inicia e salva o PID
    python /home/user/Dev/deepface-totem/compare_embeddings.py &
    echo $! > $PID_FILE
    sleep 15
fi

# Abre o navegador
chromium-browser --kiosk /home/user/Dev/deepface-totem/template/splash.html