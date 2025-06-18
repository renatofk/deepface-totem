FROM python:3.12-slim-bookworm

# Instale dependências do sistema para bibliotecas nativas
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libavdevice-dev \
    libavfilter-dev \
    libopus-dev \
    libvpx-dev \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    git \
    && apt-get clean

# Crie diretório de trabalho
WORKDIR /app

# Copie os arquivos do projeto
COPY . /app

# Instale as dependências do Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Exponha a porta usada pelo Flask
EXPOSE 5000

# Comando para rodar o app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
