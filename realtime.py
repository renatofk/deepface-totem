# from deepface import DeepFace

# DeepFace.stream(db_path='photobase',enable_face_analysis=False, detector_backend='yolov11n',model_name='Dlib')

from deepface import DeepFace


# # Executa o stream e personaliza o que fazer após reconhecimento
# DeepFace.stream(
#     db_path="photobase",
#     model_name="Dlib",
#     detector_backend="yolov11n",
#     distance_metric="euclidean_l2",
#     enable_face_analysis=False,
#     time_threshold=5,               # Evita repetir a mesma pessoa toda hora
#     frame_threshold=5              # Nº de frames para confirmar identidade
# )
# Executa o stream e personaliza o que fazer após reconhecimento
DeepFace.stream(
    db_path="photobase",
    model_name="VGG-Face",
    detector_backend="opencv",
    enable_face_analysis=False,
    time_threshold=5,               # Evita repetir a mesma pessoa toda hora
    frame_threshold=5              # Nº de frames para confirmar identidade
)
